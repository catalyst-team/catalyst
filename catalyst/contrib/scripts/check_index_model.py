# flake8: noqa
# import argparse
# import collections
#
# import nmslib
# import numpy as np
# import pandas as pd
# import tqdm
#
#
# def build_args(parser):
#     """Constructs the command-line arguments."""
#     parser.add_argument("--in-csv", type=str, default=None)
#     parser.add_argument("--in-knn", type=str, default=None)
#
#     parser.add_argument("--in-csv-test", type=str, default=None)
#     parser.add_argument("--in-npy-test", type=str, default=None)
#     parser.add_argument("--label-column", type=str, default=None)
#
#     parser.add_argument(
#         "--knn-metric", type=str, default="l2", choices=["l2", "angulardist", "cosinesimil"],
#     )
#     parser.add_argument(
#         "-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size ",
#     )
#     parser.add_argument("-k", "--recall-at", default="1,3,5,10", type=str)
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
#     """Run ``catalyst-contrib check-index-model`` script."""
#     print("[==       Loading features       ==]")
#     test_features = np.load(args.in_npy_test, mmap_mode="r")
#     test_df = pd.read_csv(args.in_csv_test)
#
#     print("[==        Loading index         ==]")
#     index = nmslib.init(
#         method="hnsw", space=args.knn_metric, data_type=nmslib.DataType.DENSE_VECTOR,
#     )
#     index.loadIndex(args.in_knn)
#     knn_df = pd.read_csv(args.in_csv)
#
#     recalls = list(map(int, args.recall_at.split(",")))
#
#     res = collections.defaultdict(lambda: [])
#     for i in tqdm.tqdm(range(0, len(test_features), args.batch_size)):
#         batch_features = test_features[i : i + args.batch_size, :]
#         pred_ind_dist = index.knnQueryBatch(batch_features, k=max(recalls))
#         pred_inds = [x[0] for x in pred_ind_dist]
#         pred_labels = [[knn_df.iloc[x_i][args.label_column] for x_i in x] for x in pred_inds]
#         pred_labels = np.array(pred_labels)
#         true_labels = test_df[args.label_column].values[i : i + args.batch_size, None]
#         for recall_i in recalls:
#             batch_ration = pred_labels[:, :recall_i] == true_labels
#             batch_ration = (batch_ration.sum(axis=1) > 0).astype(np.int32).tolist()
#             res[recall_i].extend(batch_ration)
#
#     for recall_i2 in recalls:
#         ratio = sum(res[recall_i2]) / len(res[recall_i2]) * 100.0
#         print(
#             "[==      Recall@{recall_at:2}: {ratio:.4}%      ==]".format(
#                 recall_at=recall_i2, ratio=ratio
#             )
#         )
#
#
# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
