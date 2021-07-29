from functools import partial
from itertools import islice

from callbacks import SklearnClassifierCallback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.resnet import resnet50

from catalyst import dl
from catalyst.contrib.nn import BarlowTwinsLoss


class CifarPairTransform:
    def __init__(self, train_transform=True, pair_transform=True):
        if train_transform is True:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ]
            )
        self.pair_transform = pair_transform

    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for _, module in resnet50().named_children():
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)

        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class CustomRunner(dl.Runner):
    def handle_batch(self, batch) -> None:
        if self.is_train_loader:
            (pos_1, pos_2), targets = batch
            feature_1, out_1 = self.model(pos_1)
            _, out_2 = self.model(pos_2)
            self.batch = {
                "embeddings": feature_1,
                "out_1": out_1,
                "out_2": out_2,
                "targets": targets,
            }
        else:
            images, targets = batch
            feature, _ = self.model(images)
            self.batch = {"embeddings": feature.detach().cpu(), "targets": targets.detach().cpu()}


if __name__ == "__main__":

    # hyperparams

    feature_dim, temperature, k = 128, 0.5, 200
    batch_size, epochs, num_workers = 32, 2, 2
    save_path = ""

    # data
    train_data = torchvision.datasets.CIFAR10(
        root="data", train=True, transform=CifarPairTransform(train_transform=True), download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        transform=CifarPairTransform(train_transform=False, pair_transform=False),
        download=True,
    )

    train_data = list(islice(train_data, 100))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    callbacks = [
        dl.ControlFlowCallback(
            dl.CriterionCallback(input_key="out_1", target_key="out_2", metric_key="loss"),
            loaders="train",
        ),
        SklearnClassifierCallback(
            feautres_key="embeddings",
            targets_key="targets",
            train_loader="train",
            valid_loader="valid",
            sklearn_classifier_fn=LogisticRegression,
            sklearn_metric_fn=partial(top_k_accuracy_score, **{"k": 1}),
        ),
        dl.OptimizerCallback(metric_key="loss"),
    ]

    model = Model(feature_dim).cuda()
    criterion = BarlowTwinsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)

    runner = CustomRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders={"train": train_loader, "valid": valid_loader},
        verbose=True,
        num_epochs=epochs,
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
    )
