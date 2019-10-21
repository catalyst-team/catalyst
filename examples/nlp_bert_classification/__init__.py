# flake8: noqa
# pylint: disable=unused-import
from transformers import AdamW, WarmupLinearSchedule
from catalyst.contrib.models.nlp.bert.bert import BertModel
from catalyst.dl import registry
from .experiment import Experiment

from catalyst.dl import SupervisedRunner as Runner
registry.Model(BertModel)
registry.Optimizer(AdamW, name="TransformersAdamW")
registry.Scheduler(WarmupLinearSchedule)
