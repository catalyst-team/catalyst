import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms

from catalyst.dl import utils
from catalyst.data.reader import ImageReader
from catalyst.contrib.models.encoder import ResnetEncoder

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

IMG_SIZE = (224, 224)
IMAGENET_NORM = transforms.Normalize(
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
)


def dict_transformer(sample):
    image = sample["image"]

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = np.concatenate([np.expand_dims(image, -1)] * 3, axis=-1)
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    image = torch.from_numpy(image.astype(np.float32) / 255.).permute(2, 0, 1)
    image = IMAGENET_NORM(image)

    sample["image"] = image
    return sample


def build_args(parser):
    parser.add_argument(
        "--in-csv", type=str, dest="in_csv", help="Path to csv with photos"
    )
    parser.add_argument(
        "--img-datapath",
        type=str,
        dest="datapath",
        help="Path to photos directory"
    )
    parser.add_argument(
        "--img-col",
        type=str,
        dest="img_col",
        help="Column in table that contain image path"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        dest="img_size",
        default=224,
        help="Target size of images"
    )
    parser.add_argument(
        "--out-npy",
        type=str,
        dest="out_npy",
        required=True,
        help="Path to output `.npy` file with embedded features"
    )
    parser.add_argument(
        "--arch",
        type=str,
        dest="arch",
        default="resnet18",
        help="Neural network architecture"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        dest="pooling",
        default="GlobalAvgPool2d",
        help="Type of pooling to use"
    )
    parser.add_argument(
        "--traced-model",
        type=Path,
        dest="traced_model",
        default=None,
        help="Path to pytorch traced model"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        dest="num_workers",
        help="Count of workers for dataloader",
        default=4
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Dataloader batch size",
        default=128
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print additional information"
    )
    parser.add_argument("--seed", type=int, default=42)
    utils.boolean_flag(
        parser, "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend"
    )
    utils.boolean_flag(
        parser, "benchmark",
        default=None,
        help="Use CuDNN benchmark"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    global IMG_SIZE

    utils.set_global_seed(args.seed)
    utils.prepare_cudnn(args.deterministic, args.benchmark)

    IMG_SIZE = (args.img_size, args.img_size)

    if args.traced_model is not None:
        device = utils.get_device()
        model = torch.jit.load(str(args.traced_model), map_location=device)
    else:
        model = ResnetEncoder(arch=args.arch, pooling=args.pooling)
        model = model.eval()
        model, _, _, _, device = utils.process_components(model=model)

    images_df = pd.read_csv(args.in_csv)
    images_df = images_df.reset_index().drop("index", axis=1)
    images_df = list(images_df.to_dict("index").values())

    open_fn = ImageReader(
        input_key=args.img_col, output_key="image", datapath=args.datapath
    )

    dataloader = utils.get_loader(
        images_df,
        open_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dict_transform=dict_transformer
    )

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
