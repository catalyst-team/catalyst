import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader

from catalyst import dl

from .criterion import SSDCriterion
from .dataset import DetectionDataset
from .model import SingleShotDetector

loaders = {
    "train": DataLoader(
        DetectionDataset(
            "train.json",
            "train_images",
            transforms=albu.Compose(
                [
                    albu.Resize((300, 300)),
                    albu.Normalize(),
                    albu.HorizontalFlip(p=0.5),
                    ToTensorV2(),
                ]
            ),
        ),
        batch_size=16,
        shuffle=True,
        drop_last=True,
    ),
    "valid": DataLoader(
        DetectionDataset(
            "valid.json",
            "valid_image",
            transforms=albu.Compose([albu.Resize((300, 300)), albu.Normalize(), ToTensorV2(),]),
        ),
        batch_size=16,
        shuffle=False,
        drop_last=False,
    ),
}
model = SingleShotDetector(
    backbone="resnet18", num_classes=loaders["train"].dataset.num_classes - 1,
)
criterion = SSDCriterion(
    num_classes=loaders["train"].dataset.num_classes - 1,
    ignore_class=loaders["train"].dataset.background_class,
)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# TODO: write handle_batch

runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss",
)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=1,
    callbacks=[],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    load_best_on_end=True,
)
