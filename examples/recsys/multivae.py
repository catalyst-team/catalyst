# flake8: noqa
from typing import Dict, List

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from catalyst import dl, metrics
from catalyst.contrib.datasets import MovieLens
from catalyst.contrib.layers import Normalize
from catalyst.utils.misc import set_global_seed


def collate_fn_train(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    targets = [u_items.gt(0).to(torch.float32) for u_items in batch]
    return {"inputs": torch.stack(targets), "targets": torch.stack(targets)}


def collate_fn_valid(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    test_prop = 0.2
    targets = [u_items.gt(0).to(torch.float32) for u_items in batch]

    inputs = []
    for u_items in targets:
        num_test_items = int(test_prop * torch.count_nonzero(u_items))
        u_input_items = u_items.clone()
        idx = u_items.multinomial(num_samples=num_test_items, replacement=False)
        u_input_items[idx] = 0
        inputs.append(u_input_items)

    return {"inputs": torch.stack(inputs), "targets": torch.stack(targets)}


class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super().__init__()
        self.p_dims = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        self.encoder = nn.Sequential()
        self.encoder.add_module("normalize", Normalize())
        self.encoder.add_module("dropout", nn.Dropout(dropout))
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-2], self.q_dims[1:-1])):
            self.encoder.add_module(f"encoder_fc_{i + 1}", nn.Linear(d_in, d_out))
            self.encoder.add_module(f"encoder_tanh_{i + 1}", nn.Tanh())
        self.encoder.add_module(
            f"encoder_fc_{len(self.q_dims) - 1}",
            nn.Linear(self.q_dims[-2], self.q_dims[-1] * 2),
        )

        self.decoder = nn.Sequential()
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-2], self.p_dims[1:-1])):
            self.decoder.add_module(f"decoder_fc_{i + 1}", nn.Linear(d_in, d_out))
            self.decoder.add_module(f"decoder_tanh_{i + 1}", nn.Tanh())
        self.decoder.add_module(
            f"decoder_fc_{len(self.p_dims) - 1}",
            nn.Linear(self.p_dims[-2], self.p_dims[-1]),
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    def forward(self, x):
        z = self.encoder(x)

        mu, logvar = z[:, : self.q_dims[-1]], z[:, self.q_dims[-1] :]
        z = self.reparameterize(mu, logvar)

        z = self.decoder(z)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


class RecSysRunner(dl.Runner):
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss_ae", "loss_kld", "loss"]
        }

    def handle_batch(self, batch):
        x = batch["inputs"]
        x_true = batch["targets"]
        x_recon, mu, logvar = self.model(x)

        anneal = min(
            self.hparams["anneal_cap"],
            self.batch_step / self.hparams["total_anneal_steps"],
        )

        loss_ae = -torch.mean(torch.sum(F.log_softmax(x_recon, 1) * x, -1))
        loss_kld = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        loss = loss_ae + anneal * loss_kld

        self.batch.update({"logits": x_recon, "inputs": x, "targets": x_true})

        self.batch_metrics.update(
            {"loss_ae": loss_ae, "loss_kld": loss_kld, "loss": loss}
        )
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        for key in ["loss_ae", "loss_kld", "loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


if __name__ == "__main__":
    set_global_seed(42)

    train_dataset = MovieLens(root=".", train=True, download=True)
    test_dataset = MovieLens(root=".", train=False, download=True)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_train),
        "valid": DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_valid),
    }

    item_num = len(train_dataset[0])
    model = MultiVAE([200, 600, item_num], dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    engine = dl.Engine()
    hparams = {
        "anneal_cap": 0.2,
        "total_anneal_steps": 6000,
    }
    callbacks = [
        dl.NDCGCallback("logits", "targets", [20, 50, 100]),
        dl.MAPCallback("logits", "targets", [20, 50, 100]),
        dl.MRRCallback("logits", "targets", [20, 50, 100]),
        dl.HitrateCallback("logits", "targets", [20, 50, 100]),
        dl.BackwardCallback("loss"),
        dl.OptimizerCallback("loss", accumulation_steps=1),
        dl.SchedulerCallback(),
    ]

    runner = RecSysRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        engine=engine,
        hparams=hparams,
        scheduler=lr_scheduler,
        loaders=loaders,
        num_epochs=100,
        verbose=True,
        timeit=False,
        callbacks=callbacks,
        logdir="./logs_multivae",
    )
