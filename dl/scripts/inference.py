import argparse
from pprint import pprint
from common.utils.defaults import parse_args_uargs
from common.utils.misc import set_global_seeds
from common.dl.scripts.train import prepare_modules


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--resume", default=None, type=str, metavar="PATH",
        help="path to latest checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-j", "--workers", default=None, type=int, metavar="N",
        help="number of data loading workers")
    parser.add_argument(
        "-b", "--batch-size", default=None, type=int,
        metavar="N", help="mini-batch size ")
    
    parser.add_argument("--out-prefix", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)
    pprint(args)
    pprint(config)
    set_global_seeds(args.seed)

    modules = prepare_modules(model_dir=args.model_dir)

    loaders = modules["data"].prepare_data(args, config["data_params"])
    model, _, _, _ = modules["model"].prepare_model(
        args, config)
    callbacks = modules["model"].prepare_callbacks(
        args, config, mode="infer", loggers=None)

    runner = modules["model"].ModelRunner(model=model)
    runner.infer(loaders=loaders, callbacks=callbacks)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
