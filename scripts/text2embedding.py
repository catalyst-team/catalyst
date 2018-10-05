import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from catalyst.data.reader import TextReader
from catalyst.utils.text import \
    create_fasttext_encode_fn, create_gensim_encode_fn
from catalyst.utils.defaults import create_loader
from catalyst.utils.misc import boolean_flag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", type=str)
    parser.add_argument("--fasttext-model", type=str, default=None)
    parser.add_argument("--w2v-model", type=str, default=None)
    boolean_flag(parser, "normalize", default=False)
    parser.add_argument("--txt-sep", type=str, default=" ")
    parser.add_argument("--txt-col", type=str)
    parser.add_argument(
        "--out-npy", type=str, dest="out_npy",
        required=True)
    parser.add_argument(
        "--n-workers", type=int, dest="n_workers",
        help="count of workers for dataloader", default=4)
    parser.add_argument(
        "--batch-size", type=int, dest="batch_size",
        help="dataloader batch size", default=128)
    parser.add_argument(
        "--verbose", dest="verbose",
        action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    images_df = pd.read_csv(args.in_csv)
    images_df = images_df.reset_index().drop("index", axis=1)
    images_df = list(images_df.to_dict("index").values())

    if args.fasttext_model is not None:
        encode_fn = create_fasttext_encode_fn(
            args.fasttext_model, normalize=args.normalize)
    elif args.w2v_model is not None:
        encode_fn = create_gensim_encode_fn(
            args.w2v_model, sep=args.txt_sep, normalize=args.normalize)
    else:
        raise NotImplementedError

    open_fn = TextReader(
        row_key=args.txt_col, dict_key="txt",
        encode_fn=encode_fn)

    dataloader = create_loader(
        images_df, open_fn,
        batch_size=args.batch_size,
        workers=args.n_workers)

    features = []
    dataloader = tqdm(dataloader) if args.verbose else dataloader
    for batch in dataloader:
        features_ = batch["txt"]
        features.append(features_)

    features = np.concatenate(features, axis=0)
    np.save(args.out_npy, features)


if __name__ == "__main__":
    args = parse_args()
    main(args)
