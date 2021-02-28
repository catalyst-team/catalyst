# flake8: noqa
# import argparse
# from functools import partial
# from pathlib import Path
# import sys
#
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from transformers import BertConfig, BertModel, BertTokenizer
#
# from catalyst.contrib.data import LambdaReader
# from catalyst.contrib.utils.nlp import process_bert_output, tokenize_text
# from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
# from catalyst.utils.components import process_components
# from catalyst.utils.distributed import check_ddp_wrapped
# from catalyst.utils.loaders import get_loader
# from catalyst.utils.misc import boolean_flag, set_global_seed
# from catalyst.utils.torch import any2device, prepare_cudnn
#
#
# def build_args(parser):
#     """
#     Constructs the command-line arguments for
#     ``catalyst-contrib text2embeddings``.
#
#     Args:
#         parser: parser
#
#     Returns:
#         modified parser
#     """
#     parser.add_argument("--in-csv", type=str, help="Path to csv with text", required=True)
#     parser.add_argument("--txt-col", type=str, help="Column in table that contain text")
#     parser.add_argument(
#         "--in-huggingface", type=str, required=False, help="model from huggingface hub",
#     )
#     required_path_to_model = True
#
#     for arg in sys.argv:
#         if "--in-huggingface" in arg:
#             required_path_to_model = False
#
#     if required_path_to_model:
#         parser.add_argument("--in-config", type=Path, required=required_path_to_model)
#         parser.add_argument("--in-model", type=Path, required=required_path_to_model)
#         parser.add_argument("--in-vocab", type=Path, required=required_path_to_model)
#
#     parser.add_argument(
#         "--out-prefix", type=str, required=True,
#     )
#     parser.add_argument("--max-length", type=int, default=512)  # noqa: WPS432
#     boolean_flag(parser, "mask-for-max-length", default=False)
#     boolean_flag(parser, "output-hidden-states", default=False)
#     parser.add_argument(
#         "--bert-level", type=int, help="BERT features level to use", default=None,  # noqa: WPS432
#     )
#     boolean_flag(parser, "strip", default=True)
#     boolean_flag(parser, "lowercase", default=True)
#     boolean_flag(parser, "remove-punctuation", default=True)
#     parser.add_argument("--pooling", type=str, default="avg")
#     parser.add_argument(
#         "--num-workers",
#         type=int,
#         dest="num_workers",
#         help="Count of workers for dataloader",
#         default=0,
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         dest="batch_size",
#         help="Dataloader batch size",
#         default=32,  # noqa: WPS432
#     )
#     parser.add_argument(
#         "--verbose",
#         dest="verbose",
#         action="store_true",
#         default=False,
#         help="Print additional information",
#     )
#     parser.add_argument("--seed", type=int, default=42)  # noqa: WPS432
#     boolean_flag(
#         parser,
#         "deterministic",
#         default=None,
#         help="Deterministic mode if running in CuDNN backend",
#     )
#     boolean_flag(parser, "benchmark", default=None, help="Use CuDNN benchmark")
#     boolean_flag(
#         parser, "force-save", default=None, help="Force save `.npy` with np.save",
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
# def _detach(tensor):
#     return tensor.cpu().detach().numpy()
#
#
# @torch.no_grad()
# def main(args, _=None):
#     """Run the ``catalyst-contrib text2embeddings`` script."""
#     batch_size = args.batch_size
#     num_workers = args.num_workers
#     max_length = args.max_length
#     pooling_groups = args.pooling.split(",")
#     bert_level = args.bert_level
#
#     if bert_level is not None:
#         assert args.output_hidden_states, "You need hidden states output for level specification"
#
#     set_global_seed(args.seed)
#     prepare_cudnn(args.deterministic, args.benchmark)
#
#     if getattr(args, "in_huggingface", False):
#         model_config = BertConfig.from_pretrained(args.in_huggingface)
#         model_config.output_hidden_states = args.output_hidden_states
#         model = BertModel.from_pretrained(args.in_huggingface, config=model_config)
#         tokenizer = BertTokenizer.from_pretrained(args.in_huggingface)
#     else:
#         model_config = BertConfig.from_pretrained(args.in_config)
#         model_config.output_hidden_states = args.output_hidden_states
#         model = BertModel(config=model_config)
#         tokenizer = BertTokenizer.from_pretrained(args.in_vocab)
#     if getattr(args, "in_model", None) is not None:
#         checkpoint = load_checkpoint(args.in_model)
#         checkpoint = {"model_state_dict": checkpoint}
#         unpack_checkpoint(checkpoint=checkpoint, model=model)
#
#     model = model.eval()
#     model, _, _, _, device = process_components(model=model)
#
#     df = pd.read_csv(args.in_csv)
#     df = df.dropna(subset=[args.txt_col])
#     df.to_csv(f"{args.out_prefix}.df.csv", index=False)
#     df = df.reset_index().drop("index", axis=1)
#     df = list(df.to_dict("index").values())
#     num_samples = len(df)
#
#     open_fn = LambdaReader(
#         input_key=args.txt_col,
#         output_key=None,
#         lambda_fn=partial(
#             tokenize_text,
#             strip=args.strip,
#             lowercase=args.lowercase,
#             remove_punctuation=args.remove_punctuation,
#         ),
#         tokenizer=tokenizer,
#         max_length=max_length,
#     )
#
#     dataloader = get_loader(df, open_fn, batch_size=batch_size, num_workers=num_workers)
#
#     features = {}
#     dataloader = tqdm(dataloader) if args.verbose else dataloader
#     with torch.no_grad():
#         for idx, batch_input in enumerate(dataloader):
#             batch_input = any2device(batch_input, device)
#             batch_output = model(**batch_input)
#             mask = (
#                 batch_input["attention_mask"].unsqueeze(-1) if args.mask_for_max_length else None
#             )
#
#             if check_ddp_wrapped(model):
#                 # using several gpu
#                 hidden_size = model.module.config.hidden_size
#                 hidden_states = model.module.config.output_hidden_states
#
#             else:
#                 # using cpu or one gpu
#                 hidden_size = model.config.hidden_size
#                 hidden_states = model.config.output_hidden_states
#
#             batch_features = process_bert_output(
#                 bert_output=batch_output,
#                 hidden_size=hidden_size,
#                 output_hidden_states=hidden_states,
#                 pooling_groups=pooling_groups,
#                 mask=mask,
#             )
#
#             # create storage based on network output
#             if idx == 0:
#                 for layer_name, layer_value in batch_features.items():
#                     if bert_level is not None and bert_level != layer_name:
#                         continue
#                     layer_name = layer_name if isinstance(layer_name, str) else f"{layer_name:02d}"
#                     _, embedding_size = layer_value.shape
#                     features[layer_name] = np.memmap(
#                         f"{args.out_prefix}.{layer_name}.npy",
#                         dtype=np.float32,
#                         mode="w+",
#                         shape=(num_samples, embedding_size),
#                     )
#
#             indices = np.arange(idx * batch_size, min((idx + 1) * batch_size, num_samples))
#             for layer_name2, layer_value2 in batch_features.items():
#                 if bert_level is not None and bert_level != layer_name2:
#                     continue
#                 layer_name2 = layer_name2 if isinstance(layer_name2, str) else f"{layer_name2:02d}"
#                 features[layer_name2][indices] = _detach(layer_value2)
#
#     if args.force_save:
#         for key, mmap in features.items():
#             mmap.flush()
#             np.save(f"{args.out_prefix}.{key}.force.npy", mmap, allow_pickle=False)
#
#
# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
