# flake8: noqa
from tempfile import TemporaryDirectory

from pytest import mark

from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import DATA_ROOT


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(MNIST(DATA_ROOT, train=False), batch_size=32,),
            "valid": DataLoader(MNIST(DATA_ROOT, train=False), batch_size=32,),
        }

        runner = dl.SupervisedRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss",
        )
        callbacks = [
            dl.AccuracyCallback(
                input_key="logits", target_key="targets", topk=(1, 3, 5)
            ),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
        ]
        if SETTINGS.ml_required:
            callbacks.append(
                dl.ConfusionMatrixCallback(
                    input_key="logits", target_key="targets", num_classes=10
                )
            )
        if isinstance(engine, dl.CPUEngine):
            callbacks.append(dl.AUCCallback(input_key="logits", target_key="targets"))
        # if SETTINGS.onnx_required:
        #     callbacks.append(dl.OnnxCallback(logdir=logdir, input_key="features"))
        # if SETTINGS.pruning_required:
        #     callbacks.append(
        #         dl.PruningCallback(pruning_fn="l1_unstructured", amount=0.5)
        #     )
        # if SETTINGS.quantization_required:
        #     callbacks.append(dl.QuantizationCallback(logdir=logdir))
        # if engine is None or not isinstance(engine, dl.DistributedDataParallelEngine):
        #     callbacks.append(dl.TracingCallback(logdir=logdir, input_key="features"))
        # model training
        runner.train(
            engine=engine,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=callbacks,
            logdir=logdir,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=False,
            load_best_on_end=True,
            timeit=False,
            check=False,
            overfit=False,
            fp16=False,
            ddp=False,
        )

        # loader evaluation
        metrics = runner.evaluate_loader(
            model=runner.model,
            loader=loaders["valid"],
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", topk=(1, 3, 5)
                )
            ],
        )
        assert "accuracy01" in metrics.keys()

        # model inference
        for prediction in runner.predict_loader(loader=loaders["valid"]):
            assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10
        # model post-processing
        features_batch = next(iter(loaders["valid"]))[0]
        # model stochastic weight averaging
        # model.load_state_dict(
        #     utils.get_averaged_weights_by_path_mask(logdir=logdir, path_mask="*.pth")
        # )
        # model onnx export
        if SETTINGS.onnx_required:
            utils.onnx_export(
                model=runner.model,
                batch=runner.engine.sync_device(features_batch),
                file="./mnist.onnx",
                verbose=False,
            )
        # model quantization
        if SETTINGS.quantization_required:
            utils.quantize_model(model=runner.model)
        # model pruning
        if SETTINGS.pruning_required:
            utils.prune_model(
                model=runner.model, pruning_fn="l1_unstructured", amount=0.8
            )
        # model tracing
        utils.trace_model(model=runner.model, batch=features_batch)


# Torch
def test_classification_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_classification_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_classification_on_torch_cuda1():
#     train_experiment("cuda:1")


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
)
def test_classification_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_classification_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# AMP
@mark.skipif(
    not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found"
)
def test_classification_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


@mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
    reason="No CUDA>=2 or AMP found",
)
def test_classification_on_amp_dp():
    train_experiment(dl.DataParallelEngine(fp16=True))


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_classification_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))
