# flake8: noqa
# import csv
# from pathlib import Path
# from tempfile import TemporaryDirectory

# from pytest import mark

# import torch
# from torch import nn
# from torch.optim import Adam

# from catalyst import dl
# from catalyst.contrib.datasets import MNIST
# from catalyst.contrib.losses import NTXentLoss
# from catalyst.data.dataset import SelfSupervisedDatasetWrapper
# from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
# from tests import DATA_ROOT


# def read_csv(csv_path: str):
#     with open(csv_path, "r") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=",")
#         for line_count, row in enumerate(csv_reader):
#             if line_count == 0:
#                 colnames = row
#             else:
#                 yield {colname: val for colname, val in zip(colnames, row)}


# BATCH_SIZE = 1024
# TRAIN_EPOCH = 2
# LR = 0.01
# RANDOM_STATE = 42

# if SETTINGS.ml_required:
#     from sklearn.ensemble import RandomForestClassifier

# if SETTINGS.cv_required:
#     import torchvision

#     from catalyst.contrib.data import Compose, ImageToTensor, NormalizeImage


# def train_experiment(engine=None):
#     with TemporaryDirectory() as logdir:
#         # 1. data and transforms
#         transforms = Compose(
#             [
#                 torchvision.transforms.ToPILImage(),
#                 torchvision.transforms.RandomCrop((28, 28)),
#                 torchvision.transforms.RandomVerticalFlip(),
#                 torchvision.transforms.RandomHorizontalFlip(),
#                 torchvision.transforms.ToTensor(),
#                 NormalizeImage((0.1307,), (0.3081,)),
#             ]
#         )

#         transform_original = Compose(
#             [ImageToTensor(), NormalizeImage((0.1307,), (0.3081,))]
#         )

#         mnist = MNIST(DATA_ROOT, train=True, download=True, normalize=None, numpy=True)
#         contrastive_mnist = SelfSupervisedDatasetWrapper(
#             mnist, transforms=transforms, transform_original=transform_original
#         )
#         train_loader = torch.utils.data.DataLoader(
#             contrastive_mnist, batch_size=BATCH_SIZE
#         )

#         mnist_valid = MNIST(
#             DATA_ROOT, train=False, download=True, normalize=None, numpy=True
#         )
#         contrastive_valid = SelfSupervisedDatasetWrapper(
#             mnist_valid, transforms=transforms, transform_original=transform_original
#         )
#         valid_loader = torch.utils.data.DataLoader(
#             contrastive_valid, batch_size=BATCH_SIZE
#         )

#         # 2. model and optimizer
#         encoder = nn.Sequential(
#             nn.Flatten(), nn.Linear(28 * 28, 16), nn.LeakyReLU(inplace=True)
#         )
#         projection_head = nn.Sequential(
#             nn.Linear(16, 16, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(16, 16, bias=True),
#         )

#         class ContrastiveModel(torch.nn.Module):
#             def __init__(self, model, encoder):
#                 super(ContrastiveModel, self).__init__()
#                 self.model = model
#                 self.encoder = encoder

#             def forward(self, x):
#                 emb = self.encoder(x)
#                 projection = self.model(emb)
#                 return emb, projection

#         model = ContrastiveModel(model=projection_head, encoder=encoder)

#         optimizer = Adam(model.parameters(), lr=LR)

#         # 3. criterion with triplets sampling
#         criterion = NTXentLoss(tau=0.1)

#         callbacks = [
#             dl.ControlFlowCallbackWrapper(
#                 dl.CriterionCallback(
#                     input_key="projection_left",
#                     target_key="projection_right",
#                     metric_key="loss",
#                 ),
#                 loaders="train",
#             ),
#             dl.SklearnModelCallback(
#                 feature_key="embedding_left",
#                 target_key="target",
#                 train_loader="train",
#                 valid_loaders="valid",
#                 model_fn=RandomForestClassifier,
#                 predict_method="predict_proba",
#                 predict_key="sklearn_predict",
#                 random_state=RANDOM_STATE,
#                 n_estimators=50,
#             ),
#             dl.ControlFlowCallbackWrapper(
#                 dl.AccuracyCallback(
#                     target_key="target", input_key="sklearn_predict", topk=(1, 3)
#                 ),
#                 loaders="valid",
#             ),
#         ]

#         runner = dl.SelfSupervisedRunner()

#         logdir = "./logdir"
#         runner.train(
#             model=model,
#             engine=engine,
#             criterion=criterion,
#             optimizer=optimizer,
#             callbacks=callbacks,
#             loaders={"train": train_loader, "valid": valid_loader},
#             verbose=False,
#             logdir=logdir,
#             valid_loader="train",
#             valid_metric="loss",
#             minimize_valid_metric=True,
#             num_epochs=TRAIN_EPOCH,
#         )

#         valid_path = Path(logdir) / "logs/valid.csv"
#         best_accuracy = max(
#             float(row["accuracy01"])
#             for row in read_csv(valid_path)
#             if row["accuracy01"] != "accuracy01"
#         )

#         assert best_accuracy > 0.6


# requirements_satisfied = SETTINGS.ml_required and SETTINGS.cv_required


# # Torch
# @mark.skipif(not requirements_satisfied, reason="catalyst[ml] and catalyst[cv] required")
# def test_run_on_cpu():
#     train_experiment(dl.CPUEngine())


# @mark.skipif(
#     not all([requirements_satisfied, IS_CUDA_AVAILABLE]),
#     reason="CUDA device is not available",
# )
# def test_run_on_torch_cuda0():
#     train_experiment(dl.GPUEngine())


# # @mark.skipif(
# #     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# # )
# # def test_run_on_torch_cuda1():
# #     train_experiment("cuda:1")


# @mark.skipif(
#     not all([requirements_satisfied, (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2)]),
#     reason="No CUDA>=2 found",
# )
# def test_run_on_torch_dp():
#     train_experiment(dl.DataParallelEngine())


# @mark.skipif(
#     not all([requirements_satisfied, (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2)]),
#     reason="No CUDA>=2 found",
# )
# def test_run_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# # AMP
# @mark.skipif(
#     not all([requirements_satisfied, (IS_CUDA_AVAILABLE and SETTINGS.amp_required)]),
#     reason="No CUDA or AMP found",
# )
# def test_run_on_amp():
#     train_experiment(dl.GPUEngine(fp16=True))


# @mark.skipif(
#     not all(
#         [
#             requirements_satisfied,
#             (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_amp_dp():
#     train_experiment(dl.DataParallelEngine(fp16=True))


# @mark.skipif(
#     not all(
#         [
#             requirements_satisfied,
#             (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required),
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_run_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))
