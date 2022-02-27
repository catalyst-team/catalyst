# flake8: noqa
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark

from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import (
    DATA_ROOT,
    IS_CONFIGS_REQUIRED,
    IS_CPU_REQUIRED,
    IS_DDP_AMP_REQUIRED,
    IS_DDP_REQUIRED,
    IS_DP_AMP_REQUIRED,
    IS_DP_REQUIRED,
    IS_GPU_AMP_REQUIRED,
    IS_GPU_REQUIRED,
)
from tests.misc import run_experiment_from_configs


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(DATA_ROOT, train=False),
                batch_size=32,
            ),
            "valid": DataLoader(
                MNIST(DATA_ROOT, train=False),
                batch_size=32,
            ),
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
            # fp16=False,
            # ddp=False,
        )

        if not isinstance(engine, (dl.CPUEngine, dl.GPUEngine)):
            return

        # loader evaluation
        metrics = runner.evaluate_loader(
            model=runner.model,
            engine=engine,
            loader=loaders["valid"],
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", topk=(1, 3, 5)
                )
            ],
        )
        assert "accuracy01" in metrics.keys()

        # model inference
        for prediction in runner.predict_loader(loader=loaders["valid"], engine=engine):
            assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10

        if not isinstance(engine, dl.CPUEngine):
            return

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
                batch=features_batch,
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


def train_experiment_from_configs(*auxiliary_configs: str):
    run_experiment_from_configs(
        Path(__file__).parent / "configs",
        f"{Path(__file__).stem}.yml",
        *auxiliary_configs,
    )


# Device
@mark.skipif(not IS_CPU_REQUIRED, reason="CUDA device is not available")
def test_run_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(
    not IS_CONFIGS_REQUIRED or not IS_CPU_REQUIRED, reason="CPU device is not available"
)
def test_config_run_on_cpu():
    train_experiment_from_configs("engine_cpu.yml")


@mark.skipif(
    not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason="CUDA device is not available"
)
def test_run_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


@mark.skipif(
    not IS_CONFIGS_REQUIRED or not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]),
    reason="CUDA device is not available",
)
def test_config_run_on_torch_cuda0():
    train_experiment_from_configs("engine_gpu.yml")


@mark.skipif(
    not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found",
)
def test_run_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


@mark.skipif(
    not IS_CONFIGS_REQUIRED
    or not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found",
)
def test_config_run_on_amp():
    train_experiment_from_configs("engine_gpu_amp.yml")


# DP
@mark.skipif(
    not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_run_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


@mark.skipif(
    not IS_CONFIGS_REQUIRED
    or not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_config_run_on_torch_dp():
    train_experiment_from_configs("engine_dp.yml")


@mark.skipif(
    not all(
        [
            IS_DP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_run_on_amp_dp():
    train_experiment(dl.DataParallelEngine(fp16=True))


@mark.skipif(
    not IS_CONFIGS_REQUIRED
    or not all(
        [
            IS_DP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_config_run_on_amp_dp():
    train_experiment_from_configs("engine_dp_amp.yml")


# DDP
# @mark.skipif(
#     not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
#     reason="No CUDA>=2 found",
# )
# def test_run_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# @mark.skipif(
#     not IS_CONFIGS_REQUIRED
#     or not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
#     reason="No CUDA>=2 found",
# )
# def test_config_run_on_torch_ddp():
#     train_experiment_from_configs("engine_ddp.yml")


# @mark.skipif(
#     not all(
#         [
#             IS_DDP_AMP_REQUIRED,
#             IS_CUDA_AVAILABLE,
#             NUM_CUDA_DEVICES >= 2,
#             SETTINGS.amp_required,
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))


# @mark.skipif(
#     not IS_CONFIGS_REQUIRED
#     or not all(
#         [
#             IS_DDP_AMP_REQUIRED,
#             IS_CUDA_AVAILABLE,
#             NUM_CUDA_DEVICES >= 2,
#             SETTINGS.amp_required,
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_config_run_on_amp_ddp():
#     train_experiment_from_configs("engine_ddp_amp.yml")


# def _train_fn(local_rank, world_size):
#     process_group_kwargs = {
#         "backend": "nccl",
#         "world_size": world_size,
#     }
#     os.environ["WORLD_SIZE"] = str(world_size)
#     os.environ["RANK"] = str(local_rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     dist.init_process_group(**process_group_kwargs)
#     train_experiment(dl.Engine())
#     dist.destroy_process_group()


# @mark.skipif(
#     not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
#     reason="No CUDA>=2 found",
# )
# def test_run_on_torch_ddp_spawn():
#     world_size: int = torch.cuda.device_count()
#     mp.spawn(
#         _train_fn,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )


# def _train_fn_amp(local_rank, world_size):
#     process_group_kwargs = {
#         "backend": "nccl",
#         "world_size": world_size,
#     }
#     os.environ["WORLD_SIZE"] = str(world_size)
#     os.environ["RANK"] = str(local_rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     dist.init_process_group(**process_group_kwargs)
#     train_experiment(dl.Engine(fp16=True))
#     dist.destroy_process_group()


# @mark.skipif(
#     not all(
#         [
#             IS_DDP_AMP_REQUIRED,
#             IS_CUDA_AVAILABLE,
#             NUM_CUDA_DEVICES >= 2,
#             SETTINGS.amp_required,
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_torch_ddp_amp_spawn():
#     world_size: int = torch.cuda.device_count()
#     mp.spawn(
#         _train_fn_amp,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )
#     dist.destroy_process_group()
