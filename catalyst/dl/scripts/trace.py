#!/usr/bin/env python
#
# import argparse
# from argparse import ArgumentParser
# from pathlib import Path
#
# from catalyst.utils.tracing import save_traced_model, trace_model_from_checkpoint
#
#
# def build_args(parser: ArgumentParser):
#     """Builds the command line parameters."""
#     parser.add_argument("logdir", type=Path, help="Path to model logdir")
#     parser.add_argument("--method", "-m", default="forward", help="Model method to trace")
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
#     parser.add_argument(
#         "--mode",
#         type=str,
#         choices=["eval", "train"],
#         default="eval",
#         help="Model's mode 'eval' or 'train'",
#     )
#     parser.add_argument(
#         "--with-grad",
#         action="store_true",
#         default=False,
#         help="If true, model will be traced with `requires_grad_(True)`",
#     )
#     parser.add_argument(
#         "--opt-level", type=str, default=None, help="Opt level for FP16 (optional)",
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
#         "--loader", type=str, default=None, help="Loader name to get the batch from",
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
#     """Main method for ``catalyst-dl trace``."""
#     logdir: Path = args.logdir
#     method_name: str = args.method
#     checkpoint_name: str = args.checkpoint
#     mode: str = args.mode
#     requires_grad: bool = args.with_grad
#     opt_level: str = args.opt_level
#
#     if opt_level is not None:
#         device = "cuda"
#     else:
#         device = "cpu"
#
#     traced_model = trace_model_from_checkpoint(
#         logdir,
#         method_name,
#         checkpoint_name=checkpoint_name,
#         stage=args.stage,
#         loader=args.loader,
#         mode=mode,
#         requires_grad=requires_grad,
#         opt_level=opt_level,
#         device=device,
#     )
#
#     save_traced_model(
#         model=traced_model,
#         logdir=logdir,
#         method_name=method_name,
#         mode=mode,
#         requires_grad=requires_grad,
#         opt_level=opt_level,
#         out_model=args.out_model,
#         out_dir=args.out_dir,
#         checkpoint_name=checkpoint_name,
#     )
#
#
# if __name__ == "__main__":
#     main(parse_args(), None)
