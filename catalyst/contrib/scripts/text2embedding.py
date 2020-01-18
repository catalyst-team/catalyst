import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import BertConfig, BertModel, BertTokenizer

from catalyst.contrib.modules import LamaPooling
from catalyst.data.reader import LambdaReader
from catalyst.dl import utils


def build_args(parser):
    parser.add_argument(
        "--in-csv", type=str, help="Path to csv with photos", required=True
    )
    parser.add_argument(
        "--txt-col",
        type=str,
        help="Column in table that contain text"
    )
    parser.add_argument("--in-config", type=Path, required=True)
    parser.add_argument("--in-model", type=Path, required=True)
    parser.add_argument("--in-vocab", type=Path, required=True)
    parser.add_argument(
        "--out-prefix",
        type=str,
        required=True,
    )
    parser.add_argument("--max-length", type=int, default=512)
    utils.boolean_flag(parser, "mask-for-max-length", default=False)
    utils.boolean_flag(parser, "output-hidden-states", default=False)
    parser.add_argument("--pooling", type=str, default="avg")
    parser.add_argument(
        "--num-workers",
        type=int,
        dest="num_workers",
        help="Count of workers for dataloader",
        default=0
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Dataloader batch size",
        default=32
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print additional information"
    )
    parser.add_argument("--seed", type=int, default=42)
    utils.boolean_flag(
        parser, "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend"
    )
    utils.boolean_flag(
        parser, "benchmark",
        default=None,
        help="Use CuDNN benchmark"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def get_features(
    text: str,
    tokenizer: BertTokenizer,
    max_length: int
) -> Dict[str, np.array]:
    text = text.lower()
    inputs = tokenizer.encode_plus(
        text, "",
        add_special_tokens=True,
        max_length=max_length
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    return {
        "input_ids": np.array(input_ids),
        "token_type_ids": np.array(token_type_ids),
        "attention_mask": np.array(attention_mask),
    }


def _detach(tensor):
    return tensor.cpu().detach().numpy()


def main(args, _=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_length = args.max_length
    pooling_groups = args.pooling.split(",")

    utils.set_global_seed(args.seed)
    utils.prepare_cudnn(args.deterministic, args.benchmark)

    model_config = BertConfig.from_pretrained(args.in_config)
    model_config.output_hidden_states = args.output_hidden_states
    model = BertModel(config=model_config)

    checkpoint = utils.load_checkpoint(args.in_model)
    checkpoint = {"model_state_dict": checkpoint}
    utils.unpack_checkpoint(checkpoint=checkpoint, model=model)

    model = model.eval()
    model, _, _, _, device = utils.process_components(model=model)

    tokenizer = BertTokenizer.from_pretrained(args.in_vocab)

    df = pd.read_csv(args.in_csv)
    df = df.dropna(subset=[args.txt_col])
    df.to_csv(f"{args.out_prefix}.df.csv", index=False)
    df = df.reset_index().drop("index", axis=1)
    df = list(df.to_dict("index").values())
    num_samples = len(df)

    open_fn = LambdaReader(
        input_key=args.txt_col,
        output_key=None,
        lambda_fn=get_features,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = utils.get_loader(
        df,
        open_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    features = {}
    poolings = {}
    dataloader = tqdm(dataloader) if args.verbose else dataloader
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = utils.any2device(batch, device)
            features_ = model(**batch)

            # create storage based on network output
            if idx == 0:
                # class
                _, embedding_size = features_[1].shape
                features["class"] = np.memmap(
                    f"{args.out_prefix}.class.npy",
                    dtype=np.float32,
                    mode="w+",
                    shape=(num_samples, embedding_size),
                )
                if args.output_hidden_states:
                    # all embeddings
                    for i, feature_ in enumerate(features_[2]):
                        name_ = f"embeddings_{i + 1:02d}"
                        _, _, embedding_size = feature_.shape
                        poolings[name_] = LamaPooling(
                            features_in=embedding_size,
                            groups=pooling_groups,
                        )
                        features[name_] = np.memmap(
                            f"{args.out_prefix}.{name_}.npy",
                            dtype=np.float32,
                            mode="w+",
                            shape=(num_samples, embedding_size),
                        )
                else:
                    # last
                    _, _, embedding_size = features_[0].shape
                    poolings["last"] = LamaPooling(
                        features_in=embedding_size,
                        groups=pooling_groups,
                    )
                    features["last"] = np.memmap(
                        f"{args.out_prefix}.last.npy",
                        dtype=np.float32,
                        mode="w+",
                        shape=(num_samples, embedding_size),
                    )

            indices = np.arange(
                idx * batch_size,
                min((idx + 1) * batch_size, num_samples)
            )
            features["class"][indices] = _detach(features_[1])
            mask = batch["attention_mask"].unsqueeze(-1) \
                if args.mask_for_max_length \
                else None
            if args.output_hidden_states:
                # all embeddings
                for i, feature_ in enumerate(features_[2]):
                    name_ = f"embeddings_{i + 1:02d}"
                    feature_ = poolings[name_](
                        feature_,
                        mask=mask
                    )
                    features[name_][indices] = _detach(feature_)
            else:
                feature_ = poolings[name_](
                    features_[0],
                    mask=mask
                )
                features["last"][indices] = _detach(feature_)


if __name__ == "__main__":
    args = parse_args()
    main(args)
