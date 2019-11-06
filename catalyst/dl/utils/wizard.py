import pathlib
from prompt_toolkit import prompt
from collections import OrderedDict
from catalyst.utils.scripts import import_module
from catalyst.dl import registry
import yaml

yaml.add_representer(
    OrderedDict,
    lambda dumper, data:
        dumper.represent_mapping("tag:yaml.org,2002:map", data.items()))


class Wizard():
    def __init__(self):
        print("Welcome to Catalyst Config API wizard!\n")

        self._cfg = OrderedDict([
            ("model_params", OrderedDict()),
            ("args", OrderedDict()),
            ("stages", OrderedDict())
        ])

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

    def criterion_params_step(self, stage):
        if self._skip_override_stages_common("criterion_params"):
            return
        cp = OrderedDict()
        criterions = registry.CRITERIONS.all()
        criterions = sorted([c for c in criterions if c[0].isupper()])
        msg = "What criterion you'll be using:\n\n"
        if len(criterions):
            msg += "\n".join([f"{n+1}: {c}" for n, c in enumerate(criterions)])
            print(msg)
            criterion = prompt("\nEnter number from list above or "
                               "class name of criterion you\"ll be using: ")
            if criterion.isdigit():
                criterion = criterions[int(criterion) - 1]
        else:
            criterion = prompt("Enter class name of criterion you\"ll be using: ")
        cp["criterion"] = criterion
        res = prompt("If there are arguments you want to provide during "
                     "criterion initialization, provide them here in "
                     "following format:\n\narg1=val,arg2=3.41\n\n"
                     "Or just skip this step (press Enter): ")
        if len(res):
            res = [t.split('=') for t in res.split(',')]
            for k, v in res:
                cp[k] = v
        stage["criterion_params"] = cp

    def state_params_step(self, stage):
        if self._skip_override_stages_common("state_params"):
            return
        sp = OrderedDict()
        sp["main_metric"] = prompt("What is the main_metric?: ", default="loss")
        sp["minimize_metric"] = bool(prompt("Will it be minimized (True/False): ", default="True"))
        stage["state_params"] = sp

    def data_params_step(self, stage):
        if self._skip_override_stages_common("data_params"):
            return
        dp = OrderedDict()
        dp["batch_size"] = int(prompt("What is the batch_size?: ", default="1"))
        dp["num_workers"] = int(prompt("What is the num_workers?: ", default="1"))
        stage["data_params"] = dp

    def _stage_step(self, stage):
        self.data_params_step(stage)
        self.state_params_step(stage)
        self.criterion_params_step(stage)
        return
        self.optimizer_params_step(stage)
        self.scheduler_params_step(stage)
        self.callbacks_params_step(stage)

    def stages_step(self):
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
        models = registry.MODELS.all()
        models = sorted([m for m in models if m[0].isupper()])
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
        self._cfg["args"]["expdir"] = prompt(
            "Where is the `__init__.py` with your modules stored: ",
            default="./src")
        self._cfg["args"]["logdir"] = prompt(
            "Where Catalyst supposed to save its logs: ",
            default="./logs/experiment")


def run_wizard():
    wiz = Wizard()
    wiz.args_step()
    wiz.export_user_modules()
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
