# flake8: noqa
# pylint: disable=unused-import
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.contrib.models.nlp.bert.distil_classify import (
    BertClassifier
)
from catalyst.contrib.runner.nlp.bert_supervised import (
    BertSupervisedRunner as Runner
)
from catalyst.dl import registry
from .experiment import Experiment

registry.Model(BertClassifier)
registry.Optimizer(Adam)
registry.Scheduler(ReduceLROnPlateau)
