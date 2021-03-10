# flake8: noqa

from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def train_experiment(device):
    with TemporaryDirectory() as logdir:
        # sample data
        num_users, num_features, num_items = int(1e4), int(1e1), 10
        X = torch.rand(num_users, num_features)
        y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_items)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        class CustomRunner(dl.Runner):
            def handle_batch(self, batch):
                x, y = batch
                logits = self.model(x)
                self.batch = {
                    "features": x,
                    "logits": logits,
                    "scores": torch.sigmoid(logits),
                    "targets": y,
                }

        # model training
        runner = CustomRunner()
        runner.train(
            engine=dl.DeviceEngine(device),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=1,
            verbose=False,
            callbacks=[
                dl.CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.HitrateCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.MRRCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.NDCGCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir=logdir, loader_key="valid", metric_key="map01", minimize=False
                ),
            ],
        )


def test_finetune_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_cuda():
    train_experiment("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_finetune_on_cuda_device():
    train_experiment("cuda:1")
