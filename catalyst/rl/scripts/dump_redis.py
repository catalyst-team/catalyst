#!/usr/bin/env python
import argparse
import pickle
from tqdm import tqdm
from redis import StrictRedis


def build_args(parser):
    parser.add_argument(
        "--port",
        type=int,
        default=12000)
    parser.add_argument(
        "--out-pkl",
        type=str,
        required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000)
    parser.add_argument(
        "--start-from",
        type=int,
        default=0)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    redis = StrictRedis(port=args.port)

    redis_len = redis.llen("trajectories") - 1
    episodes = []
    for i in tqdm(range(args.start_from, redis_len)):
        episode = redis.lindex("trajectories", i)
        episodes.append(episode)
        if i > args.start_from \
                and (i - args.start_from) % args.chunk_size == 0:
            with open(args.out_pkl.format(suffix=i), "wb") as fout:
                pickle.dump(episodes, fout)
            episodes = []

    with open(args.out_pkl.format(suffix=i), "wb") as fout:
        pickle.dump(episodes, fout)


if __name__ == "__main__":
    args = parse_args()
    main(args)
