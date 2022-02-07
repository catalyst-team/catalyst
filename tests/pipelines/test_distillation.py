# flake8: noqa

# import os
# from tempfile import TemporaryDirectory

# from pytest import mark

# import torch
# from torch import nn, optim
# from torch.nn import functional as F
# from torch.utils.data import DataLoader

# from catalyst import dl
# from catalyst.contrib.datasets import MNIST
# from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
# from tests import DATA_ROOT


# class DistilRunner(dl.Runner):
#     def handle_batch(self, batch):
#         x, y = batch

#         self.model["teacher"].eval()  # let's manually set teacher model to eval mode
#         with torch.no_grad():
#             t_logits = self.model["teacher"](x)

#         s_logits = self.model["student"](x)
#         self.batch = {
#             "t_logits": t_logits,
#             "s_logits": s_logits,
#             "targets": y,
#             "s_logprobs": F.log_softmax(s_logits, dim=-1),
#             "t_probs": F.softmax(t_logits, dim=-1),
#         }


# def train_experiment(engine=None):
#     with TemporaryDirectory() as logdir:
#         teacher = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
#         student = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
#         model = {"teacher": teacher, "student": student}
#         criterion = {
#             "cls": nn.CrossEntropyLoss(),
#             "kl": nn.KLDivLoss(reduction="batchmean"),
#         }
#         optimizer = optim.Adam(student.parameters(), lr=0.02)

#         loaders = {
#             "train": DataLoader(
#                 MNIST(DATA_ROOT, train=True),
#                 batch_size=32,
#             ),
#             "valid": DataLoader(
#                 MNIST(DATA_ROOT, train=False),
#                 batch_size=32,
#             ),
#         }

#         runner = DistilRunner()
#         # model training
#         runner.train(
#             engine=engine,
#             model=model,
#             criterion=criterion,
#             optimizer=optimizer,
#             loaders=loaders,
#             num_epochs=1,
#             logdir=logdir,
#             verbose=False,
#             callbacks=[
#                 dl.AccuracyCallback(
#                     input_key="t_logits",
#                     target_key="targets",
#                     num_classes=2,
#                     prefix="teacher_",
#                 ),
#                 dl.AccuracyCallback(
#                     input_key="s_logits",
#                     target_key="targets",
#                     num_classes=2,
#                     prefix="student_",
#                 ),
#                 dl.CriterionCallback(
#                     input_key="s_logits",
#                     target_key="targets",
#                     metric_key="cls_loss",
#                     criterion_key="cls",
#                 ),
#                 dl.CriterionCallback(
#                     input_key="s_logprobs",
#                     target_key="t_probs",
#                     metric_key="kl_div_loss",
#                     criterion_key="kl",
#                 ),
#                 dl.MetricAggregationCallback(
#                     metric_key="loss", metrics=["kl_div_loss", "cls_loss"], mode="mean"
#                 ),
#                 dl.BackwardCallback(metric_key="loss"),
#                 dl.OptimizerCallback(metric_key="loss"),
#                 dl.CheckpointCallback(
#                     logdir=logdir,
#                     loader_key="valid",
#                     metric_key="loss",
#                     minimize=True,
#                     topk=3,
#                 ),
#             ],
#         )


# # Torch
# def test_run_on_cpu():
#     train_experiment(dl.CPUEngine())


# @mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
# def test_run_on_torch_cuda0():
#     train_experiment(dl.GPUEngine())


# # @mark.skipif(
# #     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# # )
# # def test_run_on_torch_cuda1():
# #     train_experiment("cuda:1")


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_run_on_torch_dp():
#     train_experiment(dl.DataParallelEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_run_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# # AMP
# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and SETTINGS.amp_required), reason="No CUDA or AMP found"
# )
# def test_run_on_amp():
#     train_experiment(dl.GPUEngine(fp16=True))


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_amp_dp():
#     train_experiment(dl.DataParallelEngine(fp16=True))


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))
