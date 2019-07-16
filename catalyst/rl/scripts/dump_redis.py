#!/usr/bin/env python
import argparse
import pickle
from tqdm import tqdm
from redis import Redis

from catalyst import utils


def build_args(parser):
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--out-pkl", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--min-reward", type=int, default=None)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    db = Redis(host=args.host, port=args.port)
    redis_len = db.llen("trajectories") - 1
    trajectories = []

    for i in tqdm(range(args.start_from, redis_len)):
        trajectory = db.lindex("trajectories", i)

        if args.min_reward is not None:
            trajectory = utils.unpack(trajectory)
            if sum(trajectory["trajectory"][-2]) > args.min_reward:
                trajectory = utils.pack(trajectory)
                trajectories.append(trajectory)
        else:
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
