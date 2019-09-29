#!/usr/bin/env python
# usage:
# catalyst-rl dump-db --db=redis --out-pkl="./my_db_{suffix}.pkl"

import argparse
import pickle
from tqdm import tqdm

from catalyst import utils
from catalyst.rl.db import RedisDB, MongoDB
from catalyst.rl.utils import structed2dict_trajectory


def build_args(parser):
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--out-pkl", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--min-reward", type=int, default=None)
    parser.add_argument(
        "--db", type=str, choices=["redis", "mongo"],
        default=None, required=True)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    db_fn = RedisDB if args.db == "redis" else MongoDB
    db = db_fn(host=args.host, port=args.port)
    db_len = db.num_trajectories
    trajectories = []

    i = 0
    for i in tqdm(range(args.start_from, db_len)):
        if args.db == "redis":
            trajectory = db.get_trajectory(i)
        else:
            # mongo does not support indexing yet
            trajectory = db.get_trajectory()

        if args.min_reward is not None \
                and sum(trajectory["trajectory"][-2]) < args.min_reward:
            continue

        trajectory = structed2dict_trajectory(trajectory)
        trajectory = utils.pack(trajectory)
        trajectories.append(trajectory)

        if args.chunk_size is not None \
                and (i - args.start_from) % args.chunk_size == 0:
            with open(args.out_pkl.format(suffix=i), "wb") as fout:
                pickle.dump(trajectories, fout)
            trajectories = []

    with open(args.out_pkl.format(suffix=i), "wb") as fout:
        pickle.dump(trajectories, fout)


if __name__ == "__main__":
    args = parse_args()
    main(args)
