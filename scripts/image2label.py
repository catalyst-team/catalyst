import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch
from torchvision import models, transforms

from catalyst.utils.factory import UtilsFactory
from catalyst.data.reader import ImageReader

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


IMG_SIZE = (224, 224)
IMAGENET_NORM = transforms.Normalize(
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225))


def dict_transformer(sample):
    image = sample["image"]

    image = cv2.resize(
        image, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    image = torch.from_numpy(image.astype(np.float32) / 255.).permute(2, 0, 1)
    image = IMAGENET_NORM(image)

    sample["image"] = image
    return sample


class Images2Keywords:
    def __init__(self, nn_model, n_keywords, labels):
        self.nn_model = nn_model
        self.n_keywords = n_keywords
        self.labels = labels

    def __call__(self, images_batch):
        predict = self.nn_model(images_batch).cpu().data.numpy()

        keywords = []

        for i in predict:
            indexes = np.argsort(i)
            indexes = indexes[::-1][:self.n_keywords]

            image_keywords = []
            for index in indexes:
                keyword = self.labels[index].split(";")[0]
                image_keywords.append(keyword)

            keywords.append(";".join(image_keywords))

        return keywords


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script allow you get top-n labels for photos")
    parser.add_argument(
        "--in-csv", type=str, dest="in_csv",
        help="path to csv with photos", required=True)
    parser.add_argument(
        "--datapath", type=str, dest="datapath",
        help="path to photos directory", required=True)
    parser.add_argument(
        "--img-col", type=str, dest="img_col",
        help="column in table that contain image path", required=True)
    parser.add_argument(
        "--out-csv", type=str, dest="out_csv",
        help="output csv with keywords for every image",
        default="out.csv")
    parser.add_argument(
        "--keywords-col", type=str, dest="keywords_col",
        help="column in output csv that contain n keywords for photo",
        default="keywords")
    parser.add_argument(
        "--n-keywords", type=int, dest="n_keywords",
        help="number of keywords", default=5)
    parser.add_argument(
        "--arch", type=str, dest="arch",
        help="neural network architecture", default="resnet101")
    parser.add_argument(
        "--n-workers", type=int, dest="n_workers",
        help="count of workers for dataloader", default=4)
    parser.add_argument(
        "--batch-size", type=int, dest="batch_size",
        help="dataloader batch size", default=128)
    parser.add_argument(
        "--labels", type=str, dest="labels",
        help="json file with labels", required=True)
    parser.add_argument(
        "--verbose", dest="verbose",
        action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    model = models.__dict__[args.arch](pretrained=True)
    model = model.eval()
    model, device = UtilsFactory.prepare_model(model)

    labels = json.loads(open(args.labels).read())

    i2k = Images2Keywords(model, args.n_keywords, labels)

    images_df = pd.read_csv(args.in_csv)
    images_df = images_df.reset_index().drop("index", axis=1)
    images_df = list(images_df.to_dict("index").values())

    open_fn = ImageReader(
        row_key=args.img_col, dict_key="image",
        datapath=args.datapath)

    dataloader = UtilsFactory.create_loader(
        images_df, open_fn,
        batch_size=args.batch_size,
        workers=args.n_workers,
        dict_transform=dict_transformer)

    keywords = []
    dataloader = tqdm(dataloader) if args.verbose else dataloader
    with torch.no_grad():
        for batch in dataloader:
            keywords_batch = i2k(batch["image"].to(device))
            keywords += keywords_batch

    input_csv = pd.read_csv(args.in_csv)
    input_csv[args.keywords_col] = keywords
    input_csv.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
