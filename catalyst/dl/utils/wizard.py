from collections import OrderedDict
import pathlib

from prompt_toolkit import prompt
import yaml

from catalyst import registry
from catalyst.utils import clone_pipeline, import_module
from catalyst.utils.pipelines import URLS

yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
)


class Wizard:
    """
    Class for Catalyst Config API Wizard.

    The instance of this class will be created and called from cli command:
    ``catalyst-dl init --interactive``.

    With help of this Wizard user will be able to setup pipeline from available
    templates and make choices of what predefined classes to use in different
    parts of pipeline.
    """

    def __init__(self):
        """
        Initialization of instance of this class will print welcome message and
        logo of Catalyst in ASCII format. Also here we'll save all classes of
        Catalyst own pipeline parts to be able to put user's modules on top of
        lists to ease the choice.
        """
        self.__sep("Welcome to Catalyst Config API wizard!")
        print(
            """
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
            \n"""
        )  # noqa: WPS355

        self._cfg = OrderedDict(
            [
                ("model_params", OrderedDict()),
                ("args", OrderedDict()),
                ("stages", OrderedDict()),
            ]
        )

        self.pipeline_path = pathlib.Path("./")
        self.__before_export = {
            "MODELS": registry.__dict__["MODELS"].all(),
            "CRITERIONS": registry.__dict__["CRITERIONS"].all(),
            "OPTIMIZERS": registry.__dict__["OPTIMIZERS"].all(),
            "SCHEDULERS": registry.__dict__["SCHEDULERS"].all(),
            "CALLBACKS": registry.__dict__["CALLBACKS"].all(),
        }

    @staticmethod
    def __sep(step_name: str = None):
        """Separator between Wizard sections."""
        if step_name is None:
            print("\n" + "=" * 100 + "\n")
        else:
            msg = "\n" + "=" * 100 + "\n"
            msg += "=" * 10 + " " + step_name + " "
            msg += "=" * (100 - len(step_name) - 12)
            msg += "\n" + "=" * 100 + "\n"
            print(msg)

    @staticmethod
    def _export_step():
        print(
            "Config is complete. What is next?\n\n"
            "1. Preview config in YAML format\n"
            "2. Save config to file\n"
            "3. Discard changes and exit\n"
        )
        return prompt("Enter the number: ")

    @staticmethod
    def __res(result, is_yaml=False):
        if is_yaml:
            print(f"->\n{yaml.dump(result, default_flow_style=False)}")
        else:
            print(f"-> {result}")

    def __sorted_for_user(self, key):
        """
        Here we put user's modules of specific part of pipeline on top of
        modules predefined in Catalyst.
        """
        modules = registry.__dict__[key].all()
        user_modules = list(set(modules) - set(self.__before_export[key]))
        user_modules = sorted(user_modules)
        return user_modules + sorted(m for m in modules if m[0].isupper())

    def _preview(self):
        """Showing user final config in YAML format."""
        self.__sep()
        print(yaml.dump(self._cfg, default_flow_style=False))
        self.__sep()

    def _dump_step(self):
        """Asking where and saving final config converted into YAML."""
        path = prompt("Enter config path: ", default="./configs/config.yml")
        self.__res(path)
        path = pathlib.Path(path)
        with path.open(mode="w") as stream:
            yaml.dump(self._cfg, stream, default_flow_style=False)
        print(f"Config was written to {path}")

    def _skip_override_stages_common(self, param_name):
        """
        Stages could have common params, in that case we will ask user if it
        should be overriden for specific step. If not - we'll just skip entire
        params section for stage.
        """
        common = None
        if param_name in self._cfg["stages"]:
            common = self._cfg["stages"][param_name]
            print(
                "You have common setting for all stages:\n"
                + yaml.dump(common, default_flow_style=False)
            )
            res = prompt("Do you want to override it? (y/N): ", default="N")
            self.__res(res)
            return res.upper() == "N"
        return False

    def _callbacks_step(self, stage):
        self.__sep("Callbacks")
        print(
            "Let's add some callbacks!\n\n"
            "!!! Remember that Catalyst will add Criterion, Optimizer and "
            "Checkpoint callbacks for you\n"
            "with default settings if name of the step is NOT started "
            "with ``infer``.\n"
        )
        opts = OrderedDict()
        while True:
            callback = prompt(
                "Enter callback section name, e.g. "
                "'loss_aggregator'"
                "(or hit Enter to stop adding callbacks): "
            )
            if not callback:
                if opts:
                    stage["callbacks"] = opts
                return
            self.__res(callback)
            callback_params = OrderedDict()
            self._basic_params_step("callback", callback_params)
            opts[callback] = callback_params["callback_params"]

    def _basic_params_step(self, param, stage, optional=False) -> None:
        """
        Step #x

        Models, criterions, callbacks, schedulers could be choosen from list of
        predefined in Catalyst as well as from imported from user expdir. Also
        it even could not exist yet, so we provide a way to enter class name of
        the entity. Also we request args or params of those modules, but they
        are weak-typed now and will be all strings/ints in final config.
        """
        self.__sep(f"{param}_params")
        if self._skip_override_stages_common(f"{param}_params"):
            return
        opts = OrderedDict()
        modules = self.__sorted_for_user(f"{param.upper()}S")
        msg = f"What {param} you'll be using:\n\n"
        if modules:
            if optional:
                msg += "0: Skip this param\n"
            msg += "\n".join([f"{n+1}: {m}" for n, m in enumerate(modules)])
            print(msg)
            module = prompt(
                "\nEnter number from list above or "
                f"class name of {param} you'll be using: "
            )
            if module.isdigit():
                module = int(module)
                if module == 0:
                    self.__res("Skipping...")
                    return
                module = modules[module - 1]
                self.__res(module)
        else:
            module = prompt(
                f"Enter class name of {param} " "you'll be using: "
            )
            self.__res(module)
        opts[param] = module
        res = prompt(
            "If there are arguments you want to provide during "
            f"{param} initialization, provide them here in "
            "following format:\n\nlr=0.001,beta=3.41\n\n"
            "Or just skip this step (press Enter): "
        )
        if res:
            res = [t.split("=") for t in res.split(",")]
            for k, val in res:
                # We can add regex to parse params properly into types we need
                opts[k] = int(val) if val.isdigit() else val
        self.__res(opts, is_yaml=True)
        stage[f"{param}_params"] = opts

    def _stage_params_step(self, stage) -> None:
        """
        Step #5.b

        ``stage_params`` of Experiment.
        """
        self.__sep("stage_params")
        if self._skip_override_stages_common("stage_params"):
            return
        opts = OrderedDict()
        opts["num_epochs"] = int(
            prompt(
                "How much epochs you want to run this " "stage: ", default="1"
            )
        )
        self.__res(opts["num_epochs"])
        opts["main_metric"] = prompt(
            "What is the main_metric?: ", default="loss"
        )
        self.__res(opts["main_metric"])
        minimize = bool(
            prompt("Will it be minimized (True/False): ", default="True")
        )
        opts["minimize_metric"] = minimize
        self.__res(opts["minimize_metric"])
        stage["stage_params"] = opts

    def _data_params_step(self, stage) -> None:
        """
        Step #5.a

        Here we'll store required ``data_params``. Right now experiment
        couldn't be run without ``num_worker`` param, but it's rarely when user
        needs batch_size of 1
        """
        self.__sep("data_params")
        if self._skip_override_stages_common("data_params"):
            return
        opts = OrderedDict()
        opts["batch_size"] = int(
            prompt("What is the batch_size?: ", default="1")
        )
        self.__res(opts["batch_size"])
        opts["num_workers"] = int(
            prompt("What is the num_workers?: ", default="1")
        )
        self.__res(opts["num_workers"])
        stage["data_params"] = opts

    def _stage_step(self, stage) -> None:
        """
        Step #5

        For stages' common params and for every stage params we'll run this
        method to gather all we need to know about the stage and its settings
        """
        self._data_params_step(stage)
        self._stage_params_step(stage)
        self._basic_params_step("criterion", stage)
        self._basic_params_step("optimizer", stage)
        self._basic_params_step("scheduler", stage, optional=True)
        self._callbacks_step(stage)

    def _stages_step(self) -> None:
        """
        Step #4

        Stages params. We need to understand how much stages will be there,
        what are their names and if user wants to predefine something common
        for all stages
        """
        self.__sep("stages")
        cnt = prompt("How much stages your exepriment will contain: ")
        self.__res(cnt)
        cnt = int(cnt) or 1
        if cnt > 1:
            res = prompt(
                "Do you want to assign some common settings "
                "for all stages? (y/N): ",
                default="y",
            )
            self.__res(res)
            if res.lower() == "y":
                self._stage_step(self._cfg["stages"])
            print(f"\nNow we'll configure all {cnt} stages one-by-one\n")
        for stage_id in range(cnt):
            name = prompt(
                "What would be the name of this stage: ",
                default=f"stage{stage_id + 1}",
            )
            self.__res(name)
            stage = OrderedDict()
            self._stage_step(stage)
            self._cfg["stages"][name] = stage

    def _model_step(self):
        """
        Step #3

        We need to user choose its model for experiment
        """
        self._basic_params_step("model", self._cfg)

    def __export_user_modules(self):  # noqa: WPS112
        """
        Private method to try to export user's modules.
        We need this to add user's modules to list of choices for pipeline
        parts
        """
        try:
            # We need to import module to add possible modules to registry
            expdir = self._cfg["args"]["expdir"]
            if not isinstance(expdir, pathlib.Path):
                expdir = pathlib.Path(expdir)
            import_module(expdir)
            self.__res(f"Modules from {expdir} exported")
        except OSError:
            print(f"There is no modules to import found: {expdir}")
        except Exception as err:
            print(
                "Unexpected error when tried to import modules from "
                f"{expdir}: {err}"
            )

    def _args_step(self):
        """
        Step #2

        ``args`` section where two params required:
            expdir - where all user modules stored
            logdir - where Catalyst will write logs
        """
        self.__sep("args")
        self._cfg["args"]["expdir"] = prompt(
            "Provide expdir for your experiment "
            "(where is the `__init__.py` with your modules stored): ",
            default=str(self.pipeline_path / "src"),
        )
        self.__res(self._cfg["args"]["expdir"])
        self._cfg["args"]["logdir"] = prompt(
            "Provide logdir for your experiment "
            "(where Catalyst supposed to save its logs): ",
            default=str(self.pipeline_path / "logs/experiment"),
        )
        self.__res(self._cfg["args"]["logdir"])

        self.__export_user_modules()

    def _pipeline_step(self):
        """
        Step #1

        User can choose which pipeline to clone and if not skipped - where.
        Then pipeline will be copied in requested directory
        """
        self.__sep("Pipeline templates")
        opts = list(URLS.keys()) + ["empty"]
        opts = [opt.capitalize() for opt in opts]
        msg = "0: Skip this step\n"
        msg += "\n".join([f"{n + 1}: {v}" for n, v in enumerate(opts)])
        print(msg)
        res = int(
            prompt(
                "\nChoose pipeline template you want to init "
                "your project from: "
            )
        )
        if res == 0:
            self.__res("Skipped...")
            return
        pipeline = opts[res - 1]
        self.__res(pipeline)
        out_dir = prompt(
            f"Where we need to copy {pipeline} " "template files?: ",
            default="./",
        )
        self.pipeline_path = pathlib.Path(out_dir)
        clone_pipeline(pipeline.lower(), self.pipeline_path)
        self.__res(f"{pipeline} cloned to {self.pipeline_path}")

    def run(self):
        """Walks user through predefined wizard steps."""
        self._pipeline_step()
        self._args_step()
        self._model_step()
        self._stages_step()
        while True:
            res = self._export_step()
            if res == "1":
                self._preview()
            elif res == "2":
                self._dump_step()
                return
            elif res == "3":
                return
            else:
                print(f"Unknown option `{res}`")


def run_wizard():
    """Method to initialize and run wizard."""
    wiz = Wizard()
    wiz.run()


__all__ = ["run_wizard", "Wizard"]
