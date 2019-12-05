# flake8: noqa
# pylint: disable=unused-import

from catalyst.dl import registry

from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment

from catalyst.contrib.models.nlp.bert.distil_classify import (
    BertClassifier
)

registry.Model(BertClassifier)
