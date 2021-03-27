# flake8: noqa
# from typing import Sequence
# import argparse
# from pathlib import Path
#
# import cv2
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
#
# from catalyst.contrib.data.cv import ImageReader
# from catalyst.contrib.models.cv import ResnetEncoder
# from catalyst.utils.components import process_components
# from catalyst.utils.loaders import get_loader
# from catalyst.utils.misc import boolean_flag, set_global_seed
# from catalyst.utils.torch import get_device, prepare_cudnn
#
# IMG_SIZE = (224, 224)
#
#
# def normalize(
#     tensor: torch.Tensor,
#     mean: Sequence[float] = (0.485, 0.456, 0.406),
#     std: Sequence[float] = (0.229, 0.224, 0.225),
# ):
#     """Normalize a tensor image with mean and standard deviation.
#
#     Args:
#         tensor: Tensor image of size (C, H, W) to be normalized
#         mean: Sequence of means for each channel
#         std: Sequence of standard deviations for each channel
#
#     Returns:
#         torch.Tensor: Normalized Tensor image
#     """
#     dtype = tensor.dtype
#     mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
#     std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
#
#     if mean.ndim == 1:
#         mean = mean[:, None, None]
#     if std.ndim == 1:
#         std = std[:, None, None]
#
#     tensor.sub_(mean).div_(std)
#     return tensor
#
#
# def dict_transformer(sample):
#     """Transform wrapper."""
#     image = sample["image"]
#
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # image = np.concatenate([np.expand_dims(image, -1)] * 3, axis=-1)
#     image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
#     image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
#     image = normalize(image)
#
#     sample["image"] = image
#     return sample
#
#
# def build_args(parser):
#     """
#     Constructs the command-line arguments for
#     ``catalyst-contrib image2embeddings``.
#
#     Args:
#         parser: parser
#
#     Returns:
#         modified parser
#     """
#     parser.add_argument("--in-csv", type=str, dest="in_csv", help="Path to csv with photos")
#     parser.add_argument(
#         "--img-rootpath", type=str, dest="rootpath", help="Path to photos directory",
#     )
#     parser.add_argument(
#         "--img-col", type=str, dest="img_col", help="Column in table that contain image path",
#     )
#     parser.add_argument(
#         "--img-size", type=int, dest="img_size", default=224, help="Target size of images",
#     )
#     parser.add_argument(
#         "--out-npy",
#         type=str,
#         dest="out_npy",
#         required=True,
#         help="Path to output `.npy` file with embedded features",
#     )
#     parser.add_argument(
#         "--arch", type=str, dest="arch", default="resnet18", help="Neural network architecture",
#     )
#     parser.add_argument(
#         "--pooling",
#         type=str,
#         dest="pooling",
#         default="GlobalAvgPool2d",
#         help="Type of pooling to use",
#     )
#     parser.add_argument(
#         "--traced-model",
#         type=Path,
#         dest="traced_model",
#         default=None,
#         help="Path to pytorch traced model",
#     )
#     parser.add_argument(
#         "--num-workers",
#         type=int,
#         dest="num_workers",
#         help="Count of workers for dataloader",
#         default=0,
#     )
#     parser.add_argument(
#         "--batch-size", type=int, dest="batch_size", help="Dataloader batch size", default=32,
#     )
#     parser.add_argument(
#         "--verbose",
#         dest="verbose",
#         action="store_true",
#         default=False,
#         help="Print additional information",
#     )
#     parser.add_argument("--seed", type=int, default=42)
#     boolean_flag(
#         parser,
#         "deterministic",
#         default=None,
#         help="Deterministic mode if running in CuDNN backend",
#     )
#     boolean_flag(parser, "benchmark", default=None, help="Use CuDNN benchmark")
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
#     """Run the ``catalyst-contrib image2embeddings`` script."""
#     global IMG_SIZE
#
#     set_global_seed(args.seed)
#     prepare_cudnn(args.deterministic, args.benchmark)
#
#     IMG_SIZE = (args.img_size, args.img_size)  # noqa: WPS442
#
#     if args.traced_model is not None:
#         device = get_device()
#         model = torch.jit.load(str(args.traced_model), map_location=device)
#     else:
#         model = ResnetEncoder(arch=args.arch, pooling=args.pooling)
#         model = model.eval()
#         model, _, _, _, device = process_components(model=model)
#
#     df = pd.read_csv(args.in_csv)
#     df = df.reset_index().drop("index", axis=1)
#     df = list(df.to_dict("index").values())
#
#     open_fn = ImageReader(input_key=args.img_col, output_key="image", rootpath=args.rootpath)
#
#     dataloader = get_loader(
#         df,
#         open_fn,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         dict_transform=dict_transformer,
#     )
#
#     features = []
#     dataloader = tqdm(dataloader) if args.verbose else dataloader
#     with torch.no_grad():
#         for batch in dataloader:
#             batch_features = model(batch["image"].to(device))
#             batch_features = batch_features.cpu().detach().numpy()
#             features.append(batch_features)
#
#     features = np.concatenate(features, axis=0)
#     np.save(args.out_npy, features)
#
#
# if __name__ == "__main__":
#     cv2.setNumThreads(0)
#     cv2.ocl.setUseOpenCL(False)
#
#     args = parse_args()
#     main(args)
