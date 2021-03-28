from typing import List
import argparse
import os
from os import path

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch

from catalyst.settings import SETTINGS


def build_args(parser):
    """Constructs the command-line arguments."""
    parser.add_argument(
        "--in-npy", type=str, help="path to npy with project embeddings", required=True,
    )
    parser.add_argument("--in-csv", type=str, help="path to csv with photos", required=True)
    parser.add_argument(
        "--out-dir", type=str, default=None, help="directory to output files", required=True,
    )
    parser.add_argument(
        "--out-prefix", type=str, default=None, help="additional prefix to saved files",
    )
    parser.add_argument(
        "--img-col", type=str, default=None, help="column in the table that contains image paths",
    )
    parser.add_argument("--img-rootpath", type=str, help="path to photos directory")
    parser.add_argument(
        "--img-size",
        type=int,
        default=16,  # noqa: WPS432
        help="if --img-col is defined, "
        + "then images will be resized to (img-size, img-size, 3)",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="count of rows to use in csv " + "(if not defined then it will use whole data)",
    )
    parser.add_argument(
        "--meta-cols",
        type=str,
        default=None,
        help="columns in the table to save, separated by commas",
    )

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def _load_image_data(rootpath: str, paths: List, img_size: int):
    img_data = None

    if SETTINGS.cv_required:
        import cv2

        def _load_image(filename, size):
            image = cv2.imread(filename)[..., ::-1]
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
            return image

        image_names = [path.join(rootpath, name) for name in paths]
        img_data = np.stack([_load_image(name, img_size) for name in image_names], axis=0)
        img_data = (img_data.transpose((0, 3, 1, 2)) / 255.0).astype(np.float32)  # noqa: WPS432
        img_data = torch.from_numpy(img_data)

    return img_data


def main(args, _=None):
    """Run ``catalyst-contrib project-embeddings`` script."""
    df = pd.read_csv(args.in_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.meta_cols is not None:
        meta_header = args.meta_cols.split(",")
    else:
        meta_header = None
        # raise ValueError("meta-cols must not be None")

    features = np.load(args.in_npy, mmap_mode="r")
    assert len(df) == len(features)

    if args.num_rows is not None:
        indices = np.random.choice(len(df), args.num_rows)
        features = features[indices, :]
        df = df.iloc[indices]

    if args.img_col is not None:
        img_data = _load_image_data(
            rootpath=args.img_rootpath, paths=df[args.img_col].values, img_size=args.img_size,
        )
    else:
        img_data = None

    if meta_header is not None:
        metadata = df[meta_header].values.tolist()
        metadata = [
            [
                str(text).replace("\n", " ").replace(r"\s", " ").replace(r"\s\s+", " ").strip()
                for text in texts
            ]
            for texts in metadata
        ]
        assert len(metadata) == len(features)
    elif args.img_col is not None:

        def _image_name(s):
            splitted = s.rsplit("/", 1)
            return splitted[1] if len(splitted) else splitted[0]

        metadata = [_image_name(str(path)) for path in df[args.img_col].values]
    else:
        metadata = None

    summary_writer = SummaryWriter(args.out_dir)
    summary_writer.add_embedding(
        features,
        label_img=img_data,
        metadata=metadata,
        # metadata_header=(
        #     meta_header
        #     if meta_header is not None and len(meta_header)
        #     else None
        # ),
    )
    summary_writer.close()

    print(f"Done. Run `tensorboard --logdir={args.out_dir}` " + "to view in Tensorboard")


if __name__ == "__main__":
    args = parse_args()
    main(args)
