#!/usr/bin/env python
import argparse
import pickle
from tqdm import tqdm
from redis import Redis


def build_args(parser):
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--out-pkl", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--start-from", type=int, default=0)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    db = Redis(host=args.host, port=args.port)

    # t2 = list(map(utils.unpack, trajectories))
    # t3 = list(filter(lambda x: sum(x["trajectory"][-2]) > 50, t2))
    # t4 = list(map(utils.pack, t3))

    redis_len = db.llen("trajectories") - 1
    trajectories = []
    for i in tqdm(range(args.start_from, redis_len)):
        trajectory = db.lindex("trajectories", i)
        trajectories.append(trajectory)
        if i > args.start_from \
                and (i - args.start_from) % args.chunk_size == 0:
            with open(args.out_pkl.format(suffix=i), "wb") as fout:
                pickle.dump(trajectories, fout)
            trajectories = []

    with open(args.out_pkl.format(suffix=i), "wb") as fout:
        pickle.dump(trajectories, fout)


if __name__ == "__main__":
    args = parse_args()
    main(args)
