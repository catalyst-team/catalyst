#!/usr/bin/env python

import argparse
from glob import glob
import pandas as pd
from catalyst import utils

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.style.use("ggplot")
import seaborn as sns  # noqa: E402

sns.set(color_codes=True)


def plot_report(report, y_key, filename):
    ymin = report[y_key].min() - report[y_key].std()
    ymax = report[y_key].max() + report[y_key].std()
    name_len = len(report.iloc[0]["exp_name"]) // 5  # @TODO: hack

    plt.figure()
    report.plot(
        x="exp_name",
        y=y_key,
        kind="bar",
        legend=False,
        sort_columns=True,
        title=y_key,
        ylim=(ymin, ymax),
        figsize=(21, 9 + name_len),
        fontsize=20
    )
    plt.tight_layout()
    plt.savefig(filename, format="png", dpi=300)


def report_by_dir(folder):
    checkpoint = f"{folder}/best.pth"
    checkpoint = utils.load_checkpoint(checkpoint)
    exp_name = folder.rsplit("/", 1)[-1]
    row = {"exp_name": exp_name, "epoch": checkpoint["epoch"]}
    row.update(checkpoint["valid_metrics"])
    return row


def build_args(parser):
    parser.add_argument("--in-logdir", type=str, required=True)
    parser.add_argument("--out-logdir", type=str, required=True)
    parser.add_argument(
        "--keys", type=str, default="loss,_timers/fps,epoch"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    logdir_in = args.in_logdir
    logdir_out = args.out_logdir
    keys = args.keys.split(",")

    report = []
    for folder in sorted(glob(logdir_in)):
        try:
            row = report_by_dir(folder=folder)
            report.append(row)
        except Exception:
            continue

    report = pd.DataFrame(report)

    for y_key in keys:
        filename = f"{logdir_out}/{y_key.replace('/', '_')}.png"
        plot_report(report, y_key=y_key, filename=filename)


if __name__ == "__main__":
    args = parse_args()
    main(args, None)
