# flake8: noqa
# import argparse
# import pickle  # noqa: S403
#
# import nmslib
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import Normalizer, StandardScaler
#
#
# def build_args(parser):
#     """Constructs the command-line arguments."""
#     parser.add_argument("--in-npy", type=str, default=None)
#
#     parser.add_argument("--n-hidden", type=int, default=None)
#     parser.add_argument(
#         "--knn-metric", type=str, default="l2", choices=["l2", "angulardist", "cosinesimil"],
#     )
#
#     parser.add_argument("--out-npy", type=str, default=None)
#     parser.add_argument("--out-pipeline", type=str, default=None)
#     parser.add_argument("--out-knn", type=str, default=None)
#
#     parser.add_argument("--in-npy-test", type=str, default=None)
#     parser.add_argument("--out-npy-test", type=str, default=None)
#
#     return parser
#
#
# def parse_args():
#     """Parses the command line arguments for the main method."""
#     parser = argparse.ArgumentParser()
#     build_args(parser)
#     args = parser.parse_args()
#     return args
#
#
# def main(args, _=None):
#     """Run ``catalyst-contrib create-index-model`` script."""
#     print("[==       Loading features       ==]")
#     features = None
#     for in_npy in args.in_npy.split(","):
#         features_fold = np.load(in_npy, mmap_mode="r")
#         if features is None:
#             features = features_fold
#         else:
#             features = np.concatenate((features, features_fold), axis=0)
#
#     if args.n_hidden is not None:
#         pipeline = Pipeline(
#             [
#                 ("scale", StandardScaler()),
#                 ("pca", PCA(n_components=args.n_hidden, random_state=42)),
#                 ("normalize", Normalizer()),
#             ]
#         )
#
#         print("[==     Transforming features    ==]")
#         features = pipeline.fit_transform(features)
#         np.save(args.out_npy, features)
#
#         print(
#             "[ Explained variance ratio: {ratio:.4} ]".format(
#                 ratio=pipeline.named_steps["pca"].explained_variance_ratio_.sum()
#             )
#         )
#
#         print("[==        Saving pipeline       ==]")
#         pickle.dump(pipeline, open(args.out_pipeline, "wb"))
#
#     index = nmslib.init(
#         method="hnsw", space=args.knn_metric, data_type=nmslib.DataType.DENSE_VECTOR,
#     )
#     print("[==  Adding features to indexer  ==]")
#     index.addDataPointBatch(features)
#
#     print("[==        Creating index        ==]")
#     index.createIndex({"post": 1}, print_progress=True)
#     print("")
#     print("[==         Saving index         ==]")
#     index.saveIndex(args.out_knn)
#
#     if args.in_npy_test is not None:
#         test_features = np.load(args.in_npy_test, mmap_mode="r")
#         test_features = pipeline.transform(test_features)
#         np.save(args.out_npy_test, test_features)
#
#
# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
