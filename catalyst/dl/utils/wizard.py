import pathlib
from prompt_toolkit import prompt
from collections import OrderedDict
from catalyst.utils.scripts import import_module
from catalyst.dl import registry
from catalyst.dl.utils import clone_pipeline
import yaml

yaml.add_representer(
    OrderedDict,
    lambda dumper, data:
        dumper.represent_mapping("tag:yaml.org,2002:map", data.items()))


class Wizard():
    def __init__(self):
        self.sep("Welcome to Catalyst Config API wizard!")
        self.print_logo()

        self._cfg = OrderedDict([
            ("model_params", OrderedDict()),
            ("args", OrderedDict()),
            ("stages", OrderedDict())
        ])

        self.pipeline_path = pathlib.Path('./')
        self.__before_export = {
            "MODELS": registry.__dict__["MODELS"].all(),
            "CRITERIONS": registry.__dict__["CRITERIONS"].all(),
            "OPTIMIZERS": registry.__dict__["OPTIMIZERS"].all(),
            "SCHEDULERS": registry.__dict__["SCHEDULERS"].all()
        }

    def __sorted_for_user(self, key):
        modules = registry.__dict__[key].all()
        user_modules = list(set(modules) - set(self.__before_export[key]))
        user_modules = sorted(user_modules)
        return user_modules + sorted([m for m in modules if m[0].isupper()])

    def print_logo(self):
        print('''
                       ___________
                      (_         _)
                        |       |
                        |       |
                        |       |
                       /         \\
                      /    (      \\
                     /    /  (#    \\
                    /    #     #    \\
                   /    #       #    \\
                  /      #######      \\
                 (_____________________)
        \n''')

    def sep(self, step_name: str = None):
        if step_name is None:
            print("\n" + "="*100 + "\n")
        else:
            msg = "\n" + "="*100 + "\n"
            msg += "="*10 + " " + step_name + " "
            msg += "="*(100 - len(step_name) - 12)
            msg += "\n" + "="*100 + "\n"
            print(msg)


    def preview(self):
        print(yaml.dump(self._cfg, default_flow_style=False))

    def dump_step(self):
        path = prompt("Enter config path: ", default="./configs/config.yml")
        path = pathlib.Path(path)
        with path.open(mode="w") as stream:
            yaml.dump(self._cfg, stream, default_flow_style=False)
        print(f"Config was written to {path}")

    def export_step(self):
        print("Config is complete. What is next?\n\n"
              "1. Preview config in YAML format\n"
              "2. Save config to file\n"
              "3. Discard changes and exit\n")
        return prompt("Enter the number: ")

    def _skip_override_stages_common(self, param_name):
        common = None
        if param_name in self._cfg['stages']:
            common = self._cfg['stages'][param_name]
            print("You have common setting for all stages:\n" +
                  yaml.dump(common, default_flow_style=False))
            res = prompt("Do you want to override it? (y/N): ",
                         default="N")
            return res.upper() == 'N'
        else:
            return False

    def _basic_params_step(self, param, stage, optional=False):
        self.sep(f"stage: {param}_params")
        if self._skip_override_stages_common(f"{param}_params"):
            return
        op = OrderedDict()
        modules = self.__sorted_for_user(f"{param.upper()}S")
        msg = f"What {param} you'll be using:\n\n"
        if len(modules):
            if optional:
                msg += "0: Skip this param\n"
            msg += "\n".join([f"{n+1}: {m}" for n, m in enumerate(modules)])
            print(msg)
            module = prompt("\nEnter number from list above or "
                               f"class name of {param} you'll be using: ")
            if module.isdigit():
                module = int(module)
                if module == 0:
                    return
                module = modules[module - 1]
        else:
            module = prompt(f"Enter class name of {param} "
                               "you'll be using: ")
        op["module"] = module
        res = prompt("If there are arguments you want to provide during "
                     f"{param} initialization, provide them here in "
                     "following format:\n\nlr=0.001,beta=3.41\n\n"
                     "Or just skip this step (press Enter): ")
        if len(res):
            res = [t.split('=') for t in res.split(',')]
            for k, v in res:
                # We can add regex to parse params properly into types we need
                op[k] = int(v) if v.isdigit() else v
        stage[f"{param}_params"] = op

    def state_params_step(self, stage):
        self.sep(f"stage: state_params")
        if self._skip_override_stages_common("state_params"):
            return
        sp = OrderedDict()
        sp["main_metric"] = prompt("What is the main_metric?: ", default="loss")
        sp["minimize_metric"] = bool(prompt("Will it be minimized (True/False): ", default="True"))
        stage["state_params"] = sp

    def data_params_step(self, stage):
        self.sep(f"stage: data_params")
        if self._skip_override_stages_common("data_params"):
            return
        dp = OrderedDict()
        dp["batch_size"] = int(prompt("What is the batch_size?: ", default="1"))
        dp["num_workers"] = int(prompt("What is the num_workers?: ", default="1"))
        stage["data_params"] = dp

    def _stage_step(self, stage):
        self.data_params_step(stage)
        self.state_params_step(stage)
        self._basic_params_step("criterion", stage)
        self._basic_params_step("optimizer", stage)
        self._basic_params_step("scheduler", stage, optional=True)
        return
        self.callback_params_step(stage)

    def stages_step(self):
        self.sep("stages")
        cnt = prompt("How much stages your exepriment will contain: ")
        cnt = int(cnt) or 1
        if cnt > 1:
            res = prompt("Do you want to assign some common settings "
                         "for all stages? (y/N): ", default="y")
            if res.lower() == 'y':
                self._stage_step(self._cfg['stages'])
            print(f"\nNow we'll configure all {cnt} stages one-by-one\n")
        for stage_id in range(cnt):
            name = prompt("What would be the name of this stage: ",
                          default=f"stage{stage_id + 1}")
            stage = OrderedDict()
            self._stage_step(stage)
            self._cfg['stages'][name] = stage

    def model_step(self):
        self.sep("model")
        models = self.__sorted_for_user("MODELS")
        msg = "What model you'll be using:\n\n"
        if len(models):
            msg += "\n".join([f"{n+1}: {m}" for n, m in enumerate(models)])
            print(msg)
            model = prompt("\nEnter number from list above or "
                           "class name of model you\"ll be using: ")
            if model.isdigit():
                model = models[int(model) - 1]
        else:
            model = prompt("Enter class name of model you\"ll be using: ")

        self._cfg["model_params"]["model"] = model

    def export_user_modules(self):
        try:
            # We need to import module to add possible modules to registry
            expdir = self._cfg["args"]["expdir"]
            if not isinstance(expdir, pathlib.Path):
                expdir = pathlib.Path(expdir)
            import_module(expdir)
        except Exception as e:
            print(f"No modules were imported from {expdir}:\n{e}")

    def args_step(self):
        self.sep("args")
        self._cfg["args"]["expdir"] = prompt(
            "Where is the `__init__.py` with your modules stored: ",
            default=str(self.pipeline_path/"src"))
        self._cfg["args"]["logdir"] = prompt(
            "Where Catalyst supposed to save its logs: ",
            default=str(self.pipeline_path/"logs/experiment"))

        self.export_user_modules()

    def pipeline_step(self):
        self.sep("Pipeline templates")
        opts = ["Classification", "Segmentation", "Detection", "Empty"]
        msg = "0: Skip this step\n"
        msg += "\n".join([f"{n + 1}: {v}" for n, v in enumerate(opts)])
        print(msg)
        res = int(prompt("\nChoose pipeline template you want to init "
                         "your project from: "))
        if res == 0:
            return
        pipeline = opts[res - 1]
        out_dir = prompt(f"Where we need to copy {pipeline} "
                         "template files?: ", default="./")
        self.pipeline_path = pathlib.Path(out_dir)
        clone_pipeline(pipeline.lower(), self.pipeline_path)


def run_wizard():
    wiz = Wizard()
    wiz.pipeline_step()
    wiz.args_step()
    wiz.model_step()
    wiz.stages_step()
    while True:
        res = wiz.export_step()
        if res == "1":
            wiz.preview()
        elif res == "2":
            wiz.dump_step()
            return
        elif res == "3":
            return
        else:
            print(f"Unknown option `{res}`")
