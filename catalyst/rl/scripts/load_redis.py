import argparse
import pickle
from tqdm import tqdm
from redis import Redis


def build_args(parser):
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--in-pkl", type=str, required=True)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    db = Redis(host=args.host, port=args.port)

    trajectories = pickle.load(open(args.in_pkl, "rb"))

    for i in tqdm(range(len(trajectories))):
        db.rpush("trajectories", trajectories[i])


if __name__ == "__main__":
    args = parse_args()
    main(args)
