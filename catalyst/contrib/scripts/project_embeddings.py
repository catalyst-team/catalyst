import argparse
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins.projector import \
    visualize_embeddings, ProjectorConfig
from os import path


def build_args(parser):
    parser.add_argument(
        "--in-npy",
        type=str,
        dest="in_npy",
        help="path to npy with project embeddings",
        required=True
    )
    parser.add_argument(
        "--in-csv",
        type=str,
        dest="in_csv",
        help="path to csv with photos",
        required=True
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        dest="out_dir",
        default=None,
        help="directory to output files",
        required=True
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        dest="out_prefix",
        default=None,
        help="additional prefix to saved files"
    )

    parser.add_argument(
        "--img-col",
        type=str,
        dest="img_col",
        default=None,
        help="column in the table that contains image paths"
    )
    parser.add_argument(
        "--img-datapath",
        type=str,
        dest="img_datapath",
        help="path to photos directory"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        dest="img_size",
        default=16,
        help="if --img-col is defined, "
             "then images will be resized to (img-size, img-size, 3)"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        dest="n_rows",
        default=None,
        help="count of rows to use in csv "
             "(if not defined then it will use whole data)"
    )
    parser.add_argument(
        "--meta-cols",
        type=str,
        dest="meta_cols",
        default=None,
        help="columns in the table to save, separated by commas"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n**2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0), ) * (data.ndim - 3)
    data = np.pad(data, padding, mode="constant", constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
    )
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:]
    )
    data = (data * 255).astype(np.uint8)
    return data


def main(args, _=None):
    df = pd.read_csv(args.in_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    meta_file = (
        f"{args.out_prefix}_meta.tsv"
        if args.out_prefix is not None else "meta.tsv"
    )
    out_meta_file = path.join(args.out_dir, meta_file)
    args.meta_cols = (
        None if args.meta_cols is None else args.meta_cols.split(",")
    )
    df_meta = (df if args.meta_cols is None else df[args.meta_cols])

    df_meta.to_csv(
        out_meta_file, sep="\t", index=False, header=len(df_meta.columns) > 1
    )

    features = np.load(args.in_npy, mmap_mode="r")

    if args.n_rows is not None:
        rows_ids = np.random.choice(
            np.arange(0, len(features)), size=args.n_rows
        )
        features = features[rows_ids, :]
        df = df.iloc[rows_ids]

    if args.img_col is not None:
        img_data = np.concatenate(
            list(map(
                lambda x: np.expand_dims(
                    cv2.resize(
                        cv2.imread(path.join(args.img_datapath, x)),
                        (args.img_size, args.img_size),
                        interpolation=cv2.INTER_NEAREST),
                    0),
                df[args.img_col].values
            )),
            axis=0)
        img_data = np.array(img_data).reshape(
            -1, args.img_size, args.img_size, 3
        ).astype(np.float32)
        sprite = images_to_sprite(img_data)
        cv2.imwrite(path.join(args.out_dir, "sprite.png"), sprite)

    print(
        f"Building Tensorboard Projector metadata "
        f"for ({len(features)}) vectors: {out_meta_file}"
    )

    print("Running Tensorflow Session...")
    sess = tf.InteractiveSession()
    name = args.out_prefix or "tensors"
    tf.Variable(features, trainable=False, name=name)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(args.out_dir, sess.graph)

    # Link the embeddings into the config
    config = ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = name
    embed.metadata_path = meta_file

    if args.img_col is not None:
        # embed.sprite.image_path = path.join(args.out_dir, "sprite.png")
        embed.sprite.image_path = "sprite.png"
        embed.sprite.single_image_dim.extend(
            [img_data.shape[1], img_data.shape[1]]
        )

    # Tell the projector about the configured embeddings and metadata file
    visualize_embeddings(writer, config)

    # Save session and print run command to the output
    print("Saving Tensorboard Session...")
    saver.save(sess, path.join(args.out_dir, f"{name}.ckpt"))
    print(
        f"Done. Run `tensorboard --logdir={args.out_dir}` "
        f"to view in Tensorboard"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
