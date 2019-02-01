from typing import Type, Union, Callable, List
import copy
import torch
import torch.nn as nn
from catalyst.contrib import criterion, models, modules, optimizers
from catalyst.dl import callbacks
from catalyst.rl import agents, environments
from catalyst.rl.offpolicy import algorithms
from catalyst.dl.fp16 import Fp16Wrap

Factory = Union[Type, Callable]

_REGISTERS = {
    "agent": agents.__dict__,
    "algorithm": algorithms.__dict__,
    "callback": callbacks.__dict__,
    "criterion": criterion.__dict__,
    "environment": environments.__dict__,
    "model": models.__dict__,
    "module": modules.__dict__,
    "optimizer": optimizers.__dict__,
}


class Registry:
    @staticmethod
    def _inner_register(
        register_type: str, *object_factories: Factory
    ) -> Union[Factory, List[Factory]]:

        for factory in object_factories:
            registers = _REGISTERS[register_type]
            registers[factory.__name__] = factory

        if len(object_factories) == 1:
            return object_factories[0]
        return object_factories

    @staticmethod
    def agent(*agent_factories: Factory) -> Union[Factory, List[Factory]]:
        """Add agent type or factory method to global
            agent list to make it available in config
            Can be called or used as decorator
            :param: agent_factories
                Required agent factory (method or type)
            :returns: single agent factory or list of them
        """
        return Registry._inner_register("agent", *agent_factories)

    @staticmethod
    def algorithm(
        *algorithm_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add algorithm type or factory method to global
            algorithm list to make it available in config
            Can be called or used as decorator
            :param: algorithm_factories
                Required algorithm factory (method or type)
            :returns: single algorithm factory or list of them
        """
        return Registry._inner_register("algorithm", *algorithm_factories)

    @staticmethod
    def callback(
        *callback_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add callback type or factory method to global
            callback list to make it available in config
            Can be called or used as decorator
            :param: callback_factories
                Required callback factory (method or type)
            :returns: single callback factory or list of them
        """
        return Registry._inner_register("callback", *callback_factories)

    @staticmethod
    def criterion(
        *criterion_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add criterion type or factory method to global
            criterion list to make it available in config
            Can be called or used as decorator
            :param: criterion_factories
                Required criterion factory (method or type)
            :returns: single criterion factory or list of them
        """
        return Registry._inner_register("criterion", *criterion_factories)

    @staticmethod
    def environment(
        *environment_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add environment type or factory method to global
            environment list to make it available in config
            Can be called or used as decorator
            :param: environment_factories
                Required environment factory (method or type)
            :returns: single environment factory or list of them
        """
        return Registry._inner_register("environment", *environment_factories)

    @staticmethod
    def model(*models_factories: Factory) -> Union[Factory, List[Factory]]:
        """Add model type or factory method to global
            model list to make it available in config
            Can be called or used as decorator
            :param: models_factories
                Required model factory (method or type)
            :returns: single model factory or list of them
        """
        return Registry._inner_register("model", *models_factories)

    @staticmethod
    def module(*modules_factories: Factory) -> Union[Factory, List[Factory]]:
        """Add module type or factory method to global
            module list to make it available in config
            Can be called or used as decorator
            :param: modules_factories
                Required module factory (method or type)
            :returns: single module factory or list of them
        """
        return Registry._inner_register("module", *modules_factories)

    @staticmethod
    def optimizer(
        *optimizer_factories: Factory
    ) -> Union[Factory, List[Factory]]:
        """Add optimizer type or factory method to global
            optimizer list to make it available in config
            Can be called or used as decorator
            :param: optimizer_factories
                Required optimizer factory (method or type)
            :returns: single optimizer factory or list of them
        """
        return Registry._inner_register("optimizer", *optimizer_factories)

    @staticmethod
    def name2nn(name):
        if name is None:
            return None
        elif isinstance(name, nn.Module):
            return name
        elif isinstance(name, str):
            return _REGISTERS["module"][name]
        else:
            return name

    @staticmethod
    def get_fn(register_type: str, name: str):
        return _REGISTERS[register_type][name]

    @staticmethod
    def get_agent(agent=None, **agent_params):
        if agent is None:
            return None
        agent_fn = _REGISTERS["agent"][agent]
        try:
            agent = agent_fn(**agent_params)
        except Exception:
            agent = agent_fn.create_from_params(**agent_params)
        return agent

    @staticmethod
    def get_algorithm(algorithm=None, **algorithm_params):
        if algorithm is None:
            return None
        algorithm_fn = _REGISTERS["algorithm"][algorithm]
        try:
            algorithm = algorithm_fn(**algorithm_params)
        except Exception:
            algorithm = algorithm_fn.create_from_params(**algorithm_params)
        return algorithm

    @staticmethod
    def get_callback(callback=None, **callback_params):
        if callback is None:
            return None

        callback = _REGISTERS["callback"][callback](**callback_params)
        return callback

    @staticmethod
    def get_criterion(criterion=None, **criterion_params):
        if criterion is None:
            return None

        criterion = _REGISTERS["criterion"][criterion](**criterion_params)
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        return criterion

    @staticmethod
    def get_environment(environment=None, **environment_params):
        if environment is None:
            return None
        environment_fn = _REGISTERS["algorithm"][environment]
        try:
            environment = environment_fn(**environment_params)
        except Exception:
            environment = environment_fn.create_from_params(
                **environment_params
            )
        return environment

    @staticmethod
    def get_grad_clip_fn(func=None, **grad_clip_params):
        if func is None:
            return None

        func = torch.nn.utils.__dict__[func]
        grad_clip_params = copy.deepcopy(grad_clip_params)
        grad_clip_fn = lambda parameters: func(parameters, **grad_clip_params)
        return grad_clip_fn

    @staticmethod
    def get_model(model, fp16=False, available_networks=None, **model_params):
        fp16 = fp16 and torch.cuda.is_available()

        available_networks = available_networks or {}
        available_networks = {**available_networks, **_REGISTERS["model"]}

        model = available_networks[model](**model_params)

        if fp16:
            model = Fp16Wrap(model)

        return model

    @staticmethod
    def get_optimizer(model, fp16=False, optimizer=None, **optimizer_params):
        if optimizer is None:
            return None

        master_params = list(
            filter(lambda p: p.requires_grad, model.parameters())
        )
        if fp16:
            assert torch.backends.cudnn.enabled, \
                "fp16 mode requires cudnn backend to be enabled."
            master_params = [
                param.detach().clone().float() for param in master_params
            ]
            for param in master_params:
                param.requires_grad = True

        optimizer = _REGISTERS["optimizer"][optimizer](
            master_params, **optimizer_params
        )
        return optimizer

    @staticmethod
    def get_scheduler(optimizer, scheduler=None, **scheduler_params):
        if optimizer is None or scheduler is None:
            return None
        scheduler = torch.optim.lr_scheduler.__dict__[scheduler](
            optimizer, **scheduler_params
        )
        return scheduler
