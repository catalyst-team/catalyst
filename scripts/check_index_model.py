import argparse
import numpy as np
import nmslib
import tqdm
import collections


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-npy", type=str, default=None)
    parser.add_argument("--in-knn", type=str, default=None)

    parser.add_argument(
        "--knn-metric", type=str, default="l2",
        choices=["l2", "angulardist", "cosinesimil"])
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int,
        metavar="N", help="mini-batch size ")
    parser.add_argument("-k", "--recall-at", default="1,3,5,10", type=str)

    return parser.parse_args()


def main(args):
    print("[==       Loading features       ==]")
    features = None
    for in_npy in args.in_npy.split(","):
        features_ = np.load(in_npy, mmap_mode="r")
        if features is None:
            features = features_
        else:
            features = np.concatenate((features, features_), axis=0)

    print("[==        Loading index         ==]")
    index = nmslib.init(
        method="hnsw",
        space=args.knn_metric,
        data_type=nmslib.DataType.DENSE_VECTOR)
    index.loadIndex(args.in_knn)

    recalls = list(map(int, args.recall_at.split(",")))

    res = collections.defaultdict(lambda: [])
    for i in tqdm.tqdm(range(0, len(features), args.batch_size)):
        features_ = features[i:i + args.batch_size, :]
        ind_dist_by_sample = index.knnQueryBatch(
            features_,
            k=max(recalls))
        inds = list(map(lambda x: x[0].reshape(1, -1), ind_dist_by_sample))
        inds = np.concatenate(inds, axis=0)
        ind = np.arange(i, i + len(features_)).reshape(-1, 1)
        for r_ in recalls:
            res_ = (inds[:, :r_] == ind).sum(axis=1).tolist()
            res[r_].extend(res_)

    for r_ in recalls:
        res_ = sum(res[r_]) / len(res[r_])
        print("[==      Recall@{recall_at:2}: {ratio:.4}%      ==]".format(
            recall_at=r_, ratio=res_))


if __name__ == "__main__":
    args = parse_args()
    main(args)
