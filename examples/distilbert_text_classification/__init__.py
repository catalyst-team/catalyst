# flake8: noqa
# pylint: disable=unused-import
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.contrib.models.nlp.bert.distil_classify import DistilBertForSequenceClassification
from catalyst.dl import registry
from .experiment import Experiment
from catalyst.contrib.runner.nlp.bert_supervised import BertSupervisedRunner as Runner

registry.Model(DistilBertForSequenceClassification)
registry.Optimizer(Adam)
registry.Scheduler(ReduceLROnPlateau)
