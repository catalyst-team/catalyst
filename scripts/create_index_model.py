import argparse
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import nmslib


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-npy", type=str, default=None)

    parser.add_argument("--n-hidden", type=int, default=None)
    parser.add_argument(
        "--knn-metric", type=str, default="l2",
        choices=["l2", "angulardist", "cosinesimil"])

    parser.add_argument("--out-npy", type=str, default=None)
    parser.add_argument("--out-pipeline", type=str, default=None)
    parser.add_argument("--out-knn", type=str, default=None)

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

    if args.n_hidden is not None:
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=args.n_hidden, random_state=42)),
            ("normalize", Normalizer()),
        ])

        print("[==     Transforming features    ==]")
        features = pipeline.fit_transform(features)
        np.save(args.out_npy, features)

        print("[ Explained variance ratio: {ratio:.4} ]".format(
            ratio=pipeline.named_steps["pca"].explained_variance_ratio_.sum()))

        print("[==        Saving pipeline       ==]")
        pickle.dump(pipeline, open(args.out_pipeline, "wb"))

    index = nmslib.init(
        method="hnsw",
        space=args.knn_metric,
        data_type=nmslib.DataType.DENSE_VECTOR)
    print("[==  Adding features to indexer  ==]")
    index.addDataPointBatch(features)

    print("[==        Creating index        ==]")
    index.createIndex({"post": 1}, print_progress=True)
    print("")
    print("[==         Saving index         ==]")
    index.saveIndex(args.out_knn)


if __name__ == "__main__":
    args = parse_args()
    main(args)
