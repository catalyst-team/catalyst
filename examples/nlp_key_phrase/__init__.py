# flake8: noqa
# pylint: disable=unused-import
from transformers import AdamW, WarmupLinearSchedule

from catalyst.contrib.runner.nlp.bert_supervised \
    import BertSupervisedRunner as Runner
from catalyst.contrib.models.nlp.bert.bert import BertModel
from catalyst.dl import registry
from .experiment import Experiment

registry.Model(BertModel)
registry.Optimizer(AdamW, name="TransformersAdamW")
registry.Scheduler(WarmupLinearSchedule)