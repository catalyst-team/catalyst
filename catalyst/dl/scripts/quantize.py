#!/usr/bin/env python
#
# import argparse
# from argparse import ArgumentParser
# import logging
# from pathlib import Path
#
# from catalyst.utils.quantization import quantize_model_from_checkpoint, save_quantized_model
#
#
# def build_args(parser: ArgumentParser):
#     """Builds the command line parameters."""
#     parser.add_argument("--logdir", type=Path, help="Path to model logdir")
#     parser.add_argument(
#         "--checkpoint",
#         "-c",
#         default="best",
#         help="Checkpoint's name to trace",
#         metavar="CHECKPOINT_NAME",
#     )
#     parser.add_argument(
#         "--out-dir", type=Path, default=None, help="Output directory to save traced model",
#     )
#     parser.add_argument(
#         "--out-model",
#         type=Path,
#         default=None,
#         help="Output path to save traced model (overrides --out-dir)",
#     )
#
#     parser.add_argument(
#         "--stage",
#         type=str,
#         default=None,
#         help="Stage from experiment from which model and loader will be taken",
#     )
#
#     parser.add_argument(
#         "--verbose", action="store_true",
#     )
#
#     parser.add_argument(
#         "--backend", type=str, default=None, help="Defines backend for quantization",
#     )
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
# def main(args, _):
#     """Main method for ``catalyst-dl quantize``."""
#     logdir: Path = args.logdir
#     checkpoint_name: str = args.checkpoint
#
#     if args.verbose:
#         logging.basicConfig(level=logging.INFO)
#     else:
#         logging.basicConfig(level=logging.WARNING)
#
#     quantized_model = quantize_model_from_checkpoint(
#         logdir, checkpoint_name=checkpoint_name, stage=args.stage, backend=args.backend,
#     )
#
#     save_quantized_model(
#         model=quantized_model,
#         logdir=logdir,
#         out_model=args.out_model,
#         out_dir=args.out_dir,
#         checkpoint_name=checkpoint_name,
#     )
#
#
# if __name__ == "__main__":
#     main(parse_args(), None)
