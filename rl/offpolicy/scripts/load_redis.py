import argparse
import pickle
from tqdm import tqdm
from redis import StrictRedis


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        type=int,
        default=12000)
    parser.add_argument(
        "--in-pkl",
        type=str,
        required=True)

    args = parser.parse_args()
    return args


def main(args):
    redis = StrictRedis(port=args.port)

    episodes = pickle.load(open(args.in_pkl, "rb"))

    for i in tqdm(range(len(episodes))):
        redis.rpush("trajectories", episodes[i])


if __name__ == "__main__":
    args = parse_args()
    main(args)
