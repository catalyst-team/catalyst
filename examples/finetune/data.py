import numpy as np
import collections
import cv2

import torch
from torchvision import transforms
from albumentations import (
    Resize, JpegCompression, Normalize, HorizontalFlip, ShiftScaleRotate,
    CLAHE, Blur, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise,
    MotionBlur, MedianBlur, IAASharpen, IAAEmboss, RandomContrast,
    RandomBrightness, OneOf, Compose
)

from catalyst.legacy.utils.parse import parse_in_csvs
from catalyst.dl.utils import UtilsFactory
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose
from catalyst.data.augmentor import Augmentor
from catalyst.data.sampler import BalanceClassSampler
from catalyst.dl.datasource import AbstractDataSource

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ---- Augmentations ----

IMG_SIZE = 224


def strong_aug(p=.5):
    return Compose(
        [
            Resize(IMG_SIZE, IMG_SIZE),
            Compose(
                [
                    JpegCompression(p=0.9),
                    HorizontalFlip(p=0.5),
                    OneOf([
                        IAAAdditiveGaussianNoise(),
                        GaussNoise(),
                    ], p=0.5),
                    OneOf(
                        [
                            MotionBlur(p=.2),
                            MedianBlur(blur_limit=3, p=.1),
                            Blur(blur_limit=3, p=.1),
                        ],
                        p=0.5
                    ),
                    ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.2,
                        rotate_limit=15,
                        p=.5
                    ),
                    OneOf(
                        [
                            CLAHE(clip_limit=2),
                            IAASharpen(),
                            IAAEmboss(),
                            RandomContrast(),
                            RandomBrightness(),
                        ],
                        p=0.5
                    ),
                    HueSaturationValue(p=0.5),
                ],
                p=p
            ),
            Normalize(),
        ]
    )


AUG_TRAIN = strong_aug(p=0.75)
AUG_INFER = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(),
])

TRAIN_TRANSFORM_FN = [
    Augmentor(
        dict_key="image", augment_fn=lambda x: AUG_TRAIN(image=x)["image"]
    ),
]

INFER_TRANSFORM_FN = [
    Augmentor(
        dict_key="image", augment_fn=lambda x: AUG_INFER(image=x)["image"]
    ),
    Augmentor(
        dict_key="image",
        augment_fn=lambda x: torch.tensor(x).permute(2, 0, 1)
    ),
]

# ---- Data ----


class DataSource(AbstractDataSource):
    @staticmethod
    def prepare_transforms(*, mode, stage=None, **kwargs):
        if mode == "train":
            if stage in ["debug", "stage1"]:
                return transforms.Compose(TRAIN_TRANSFORM_FN)
            elif stage == "stage2":
                return transforms.Compose(INFER_TRANSFORM_FN)
        elif mode == "valid":
            return transforms.Compose(INFER_TRANSFORM_FN)
        elif mode == "infer":
            return transforms.Compose(INFER_TRANSFORM_FN)

    @staticmethod
    def prepare_loaders(
        *,
        mode: str,
        stage: str = None,
        n_workers: int = None,
        batch_size: int = None,
        datapath=None,
        in_csv=None,
        in_csv_train=None,
        in_csv_valid=None,
        in_csv_infer=None,
        train_folds=None,
        valid_folds=None,
        tag2class=None,
        class_column=None,
        tag_column=None,
        folds_seed=42,
        n_folds=5
    ):
        loaders = collections.OrderedDict()

        df, df_train, df_valid, df_infer = parse_in_csvs(
            in_csv=in_csv,
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            folds_seed=folds_seed,
            n_folds=n_folds
        )

        open_fn = [
            ImageReader(
                row_key="filepath", dict_key="image", datapath=datapath
            ),
            ScalarReader(
                row_key="class",
                dict_key="targets",
                default_value=-1,
                dtype=np.int64
            )
        ]
        open_fn = ReaderCompose(readers=open_fn)

        if len(df_train) > 0:
            labels = [x["class"] for x in df_train]
            sampler = BalanceClassSampler(labels, mode="upsampling")

            train_loader = UtilsFactory.create_loader(
                data_source=df_train,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="train", stage=stage
                ),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=sampler is None,
                sampler=sampler
            )

            print("Train samples", len(train_loader) * batch_size)
            print("Train batches", len(train_loader))
            loaders["train"] = train_loader

        if len(df_valid) > 0:
            sampler = None

            valid_loader = UtilsFactory.create_loader(
                data_source=df_valid,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="valid", stage=stage
                ),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=sampler
            )

            print("Valid samples", len(valid_loader) * batch_size)
            print("Valid batches", len(valid_loader))
            loaders["valid"] = valid_loader

        if len(df_infer) > 0:
            infer_loader = UtilsFactory.create_loader(
                data_source=df_infer,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="infer", stage=None
                ),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=None
            )

            print("Infer samples", len(infer_loader) * batch_size)
            print("Infer batches", len(infer_loader))
            loaders["infer"] = infer_loader

        return loaders
