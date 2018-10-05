import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms

from catalyst.utils.factory import UtilsFactory
from catalyst.data.reader import ImageReader
from catalyst.models.resnet_encoder import ResnetEncoder

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-csv", type=str, dest="in_csv",
        help="path to csv with photos")
    parser.add_argument(
        "--datapath", type=str, dest="datapath",
        help="path to photos directory")
    parser.add_argument(
        "--img-col", type=str, dest="img_col",
        help="column in table that contain image path")
    parser.add_argument(
        "--img-size", type=int, dest="img_size",
        default=224)
    parser.add_argument(
        "--out-npy", type=str, dest="out_npy",
        required=True)
    parser.add_argument(
        "--arch", type=str, dest="arch",
        help="neural network architecture", default="resnet101")
    parser.add_argument(
        "--pooling", type=str, dest="pooling",
        default="GlobalAvgPool2d")
    parser.add_argument(
        "--n-workers", type=int, dest="n_workers",
        help="count of workers for dataloader", default=4)
    parser.add_argument(
        "--batch-size", type=int, dest="batch_size",
        help="dataloader batch size", default=128)
    parser.add_argument(
        "--verbose", dest="verbose",
        action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    global IMG_SIZE

    IMG_SIZE = (args.img_size, args.img_size)

    model = ResnetEncoder(arch=args.arch, pooling=args.pooling)
    model = model.eval()
    model, device = UtilsFactory.prepare_model(model)

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

    features = []
    dataloader = tqdm(dataloader) if args.verbose else dataloader
    with torch.no_grad():
        for batch in dataloader:
            features_ = model(batch["image"].to(device))
            features_ = features_.cpu().detach().numpy()
            features.append(features_)

    features = np.concatenate(features, axis=0)
    np.save(args.out_npy, features)


if __name__ == "__main__":
    args = parse_args()
    main(args)
