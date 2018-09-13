import os
import shutil
import argparse
import pathlib2
from datetime import datetime
from pprint import pprint

from common.utils.defaults import parse_args_uargs, create_loggers
from common.utils.misc import \
    create_if_need, set_global_seeds, boolean_flag, import_module


def prepare_modules(model_dir, dump_dir=None):
    model_dir = (
        model_dir[:-1]
        if model_dir.endswith("/")
        else model_dir)
    model_dir_name = model_dir.rsplit("/", 1)[-1]

    new_model_dir = None
    if dump_dir is not None:
        current_date = datetime.now().strftime('%y-%m-%d-%H-%M-%S-%M-%f')
        new_src_dir = f"/src-{current_date}/"

        new_model_dir = f"{new_src_dir}" + model_dir
        new_model_dir = dump_dir + new_model_dir
        create_if_need(new_model_dir)

        # @TODO: hardcoded
        old_common_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        new_common_dir = dump_dir + f"/{new_src_dir}/common/"
        shutil.copytree(old_common_dir, new_common_dir)

    pyfiles = list(map(
        lambda x: x.name[:-3],
        pathlib2.Path(model_dir).glob("*.py")))

    modules = {}
    for name in pyfiles:
        module_name = f"{model_dir_name}.{name}"
        module_src = model_dir + "/" + f"{name}.py"

        module = import_module(module_name, module_src)
        modules[name] = module

        if new_model_dir is not None:
            module_dst = new_model_dir + "/" + f"{name}.py"
            shutil.copy2(module_src, module_dst)

    return modules


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
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
    boolean_flag(parser, "debug", default=False)

    args, unknown_args = parser.parse_known_args()

    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args, dump_config=True)
    pprint(args)
    pprint(config)
    set_global_seeds(args.seed)

    create_if_need(args.logdir)
    modules = prepare_modules(model_dir=args.model_dir, dump_dir=args.logdir)

    loaders = modules["data"].prepare_data(args, config["data_params"])
    loggers = create_loggers(args.logdir, loaders)
    model, criterion, optimizer, scheduler = modules["model"].prepare_model(
        args, config)
    mode = "debug" if args.debug else "train"
    callbacks = modules["model"].prepare_callbacks(
        args, config, mode=mode, loggers=loggers)

    runner = modules["model"].ModelRunner(
        model=model, criterion=criterion,
        optimizer=optimizer, scheduler=scheduler)
    runner.train(loaders=loaders, callbacks=callbacks, epochs=args.epochs)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
