#!/usr/bin/env python

import pathlib
import argparse
from prompt_toolkit import prompt
from collections import OrderedDict
from catalyst.utils.scripts import import_module
from catalyst.dl import registry
import yaml

yaml.add_representer(
    OrderedDict,
    lambda dumper, data:
        dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))


class Wizard():
    def __init__(self):
        print('Welcome to Catalyst Config API wizard!\n')

        self._cfg = OrderedDict([
            ('model_params', OrderedDict()),
            ('args', OrderedDict()),
            ('stages', OrderedDict())
        ])

    def preview(self):
        print(yaml.dump(self._cfg, default_flow_style=False))

    def dump_step(self):
        path = prompt('Enter config path: ', default='./configs/config.yml')
        path = pathlib.Path(path)
        with path.open(mode='w') as stream:
            yaml.dump(self._cfg, stream, default_flow_style=False)
        print(f'Config was written to {path}')

    def export_step(self):
        print('Config is complete. What is next?\n\n'
              '1. Preview config in YAML format\n'
              '2. Save config to file\n'
              '3. Discard changes and exit\n')
        return prompt('Enter the number: ', default='1')

    def model_step(self):
        models = registry.MODELS.all()
        models = sorted([m for m in models if m[0].isupper()])
        msg = "What model you'll be using:\n\n"
        if len(models):
            msg += '\n'.join([f'{n+1}: {m}' for n, m in enumerate(models)])
            print(msg)
            model = prompt('\nEnter number from list above or '
                           'class name of model you\'ll be using: ')
            if model.isdigit():
                model = models[int(model) - 1]
        else:
            model = prompt('Enter class name of model you\'ll be using: ')

        self._cfg['model_params']['model'] = model

    def export_user_modules(self):
        try:
            # We need to import module to add possible modules to registry
            expdir = self._cfg['args']['expdir']
            if not isinstance(expdir, pathlib.Path):
                expdir = pathlib.Path(expdir)
            import_module(expdir)
        except Exception as e:
            print(f'No modules were imported from {expdir}:\n{e}')

    def args_step(self):
        self._cfg['args']['expdir'] = prompt(
            'Where is the `__init__.py` with your modules stored: ',
            default='./src')
        self._cfg['args']['logdir'] = prompt(
            'Where Catalyst supposed to save its logs: ',
            default='./logs/experiment')


def build_args(parser):
    parser.add_argument("--type", type=str, default="classification")

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    if args != 'classification':
        print('Right now only classification type of wizard implemented')
    wiz = Wizard()
    wiz.args_step()
    wiz.export_user_modules()
    wiz.model_step()
    while True:
        res = wiz.export_step()
        if res == '1':
            wiz.preview()
        elif res == '2':
            wiz.dump_step()
            return
        elif res == '3':
            return
        else:
            print(f'Unknown option `{res}`')


if __name__ == '__main__':
    args = parse_args()
    main(args, None)
