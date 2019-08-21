#!/usr/bin/env python
# usage:
# catalyst-rl load-db --db=redis --in-pkl ./my_db_0.pkl ./my_db_1.pkl

import argparse
import pickle
from tqdm import tqdm

from catalyst import utils
from catalyst.rl.db import RedisDB, MongoDB


def build_args(parser):
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument(
        "--in-pkl",
        "-P",
        nargs="+",
        metavar="PKL_PATH",
        dest="in_pkl",
        required=True)
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

    for in_pkl_ in args.in_pkl:
        with open(in_pkl_, "rb") as fin:
            trajectories = pickle.load(fin)

        for trajectory in tqdm(trajectories):
            trajectory = utils.unpack_if_needed(trajectory)
            db.put_trajectory(trajectory)


if __name__ == "__main__":
    args = parse_args()
    main(args)
