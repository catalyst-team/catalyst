# flake8: noqa

import os
from tempfile import TemporaryDirectory

from pytest import mark
from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def train_experiment(device):
    with TemporaryDirectory() as logdir:
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
            ),
        }

        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )
        # model training
        runner.train(
            engine=dl.DeviceEngine(device),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3, 5)),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=10
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
                dl.ConfusionMatrixCallback(
                    input_key="logits", target_key="targets", num_classes=10
                ),
            ],
            logdir=logdir,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
            load_best_on_end=True,
            timeit=False,
            check=False,
            overfit=False,
            fp16=False,
            ddp=False,
        )
        # model inference
        for prediction in runner.predict_loader(loader=loaders["valid"]):
            assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10

        # model post-processing
        features_batch = next(iter(loaders["valid"]))[0]
        # model tracing
        utils.trace_model(model=runner.model, batch=features_batch)
        # model to onnx
        utils.onnx_export(
            model=runner.model, batch=features_batch, file=f"./{logdir}/mnist.onnx", verbose=True
        )
        # model quantization
        utils.quantize_model(model=runner.model)
        # model pruning
        # utils.prune_model(model=runner.model, pruning_fn="l1_unstructured", amount=0.8)


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
