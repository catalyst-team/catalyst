import torch
import torch.nn as nn
from catalyst.dl.callbacks import Callback
from catalyst.dl.runner import BaseExperimentRunner
from catalyst.contrib.models import ResnetEncoder, SequentialNet
from catalyst.contrib.registry import Registry

# ---- Model ----


class MiniNet(nn.Module):
    def __init__(
        self,
        enc,
        n_cls,
        hiddens,
        emb_size,
        activation_fn=torch.nn.ReLU,
        norm_fn=None,
        bias=True,
        dropout=None
    ):
        super().__init__()
        self.encoder = enc
        self.emb_net = SequentialNet(
            hiddens=hiddens + [emb_size],
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias,
            dropout=dropout
        )
        self.head = nn.Linear(emb_size, n_cls, bias=True)

    def forward(self, *, image):
        features = self.encoder(image)
        embeddings = self.emb_net(features)
        logits = self.head(embeddings)
        return embeddings, logits


@Registry.model
def baseline(encoder_params, head_params):
    img_enc = ResnetEncoder(**encoder_params)
    net = MiniNet(enc=img_enc, **head_params)
    return net


def prepare_logdir(config):
    model_params = config["model_params"]
    data_params = config["stages"]["data_params"]
    return f"{data_params['train_folds']}" \
           f"-{model_params['model']}" \
           f"-{model_params['encoder_params']['arch']}" \
           f"-{model_params['encoder_params']['pooling']}" \
           f"-{model_params['head_params']['hiddens']}" \
           f"-{model_params['head_params']['emb_size']}"


# ---- Callbacks ----


@Registry.callback
class EmbeddingsLossCallback(Callback):
    def __init__(self, emb_l2_reg=-1):
        self.emb_l2_reg = emb_l2_reg

    def on_batch_end(self, state):
        embeddings = state.output["embeddings"]
        logits = state.output["logits"]

        loss = state.criterion(logits.float(), state.input["targets"].long())

        if self.emb_l2_reg > 0:
            emb_loss = torch.mean(torch.norm(embeddings.float(), dim=1))
            loss += emb_loss * self.emb_l2_reg

        state.loss = loss


# ---- Runner ----


class ModelRunner(BaseExperimentRunner):
    @staticmethod
    def prepare_stage_model(*, model, stage, **kwargs):
        BaseExperimentRunner.prepare_stage_model(
            model=model, stage=stage, **kwargs
        )
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            for param in model_.encoder.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in model_.encoder.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError

    @staticmethod
    def _batch_handler(*, dct, model):
        embeddings, logits = model(image=dct["image"])
        output = {"embeddings": embeddings, "logits": logits}
        return output
