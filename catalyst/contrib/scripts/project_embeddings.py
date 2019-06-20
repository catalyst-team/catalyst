import argparse
import os
from os import path

import cv2
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter


def build_args(parser):
    parser.add_argument(
        "--in-npy",
        type=str,
        help="path to npy with project embeddings",
        required=True
    )
    parser.add_argument(
        "--in-csv",
        type=str,
        help="path to csv with photos",
        required=True
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="directory to output files",
        required=True
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="additional prefix to saved files"
    )
    parser.add_argument(
        "--img-col",
        type=str,
        default=None,
        help="column in the table that contains image paths"
    )
    parser.add_argument(
        "--img-datapath",
        type=str,
        help="path to photos directory"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=16,
        help="if --img-col is defined, "
             "then images will be resized to (img-size, img-size, 3)"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="count of rows to use in csv "
             "(if not defined then it will use whole data)"
    )
    parser.add_argument(
        "--meta-cols",
        type=str,
        default=None,
        help="columns in the table to save, separated by commas"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def load_image(filename, size):
    image = cv2.imread(filename)[..., ::-1]
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    return image


def main(args, _=None):
    df = pd.read_csv(args.in_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.meta_cols is not None:
        meta_header = args.meta_cols.split(",")
    else:
        meta_header = None

    features = np.load(args.in_npy, mmap_mode="r")

    if args.n_rows is not None:
        df = df.sample(n=args.n_rows)

    if args.img_col is not None:
        image_names = [path.join(args.img_datapath, name)
                       for name in df[args.img_col].values]
        img_data = np.stack([
            load_image(name, args.img_size)
            for name in image_names],
            axis=0)
        img_data = (
            img_data.transpose((0, 3, 1, 2)) / 255.0).astype(np.float32)
        img_data = torch.from_numpy(img_data)
    else:
        img_data = None

    summary_writer = SummaryWriter(args.out_dir)
    summary_writer.add_embedding(
        features,
        metadata=df.values,
        label_img=img_data,
        metadata_header=meta_header
    )

    print(
        f"Done. Run `tensorboard --logdir={args.out_dir}` "
        f"to view in Tensorboard"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
