# flake8: noqa
from typing import Dict, List

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl, metrics
from catalyst.contrib.datasets import MovieLens
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


class MacridVAE(nn.Module):
    def __init__(self, q_dims, kfac=7, tau=0.1, nogb=False, dropout=0.5):
        super().__init__()
        self.q_dims = q_dims
        self.kfac = kfac
        self.tau = tau
        self.nogb = nogb

        self.item_embedding = nn.Embedding(self.q_dims[0], self.q_dims[-1])
        self.k_embedding = nn.Embedding(self.kfac, self.q_dims[-1])

        self.encoder = nn.Sequential()
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-2], self.q_dims[1:-1])):
            self.encoder.add_module(f"encoder_fc_{i + 1}", nn.Linear(d_in, d_out))
            self.encoder.add_module(f"encoder_tanh_{i + 1}", nn.Tanh())
        self.encoder.add_module(
            f"encoder_fc_{len(self.q_dims) - 1}",
            nn.Linear(self.q_dims[-2], self.q_dims[-1] * 2),
        )

        self.drop = nn.Dropout(dropout)

        self.encoder.apply(self.init_weights)

    def forward(self, x):
        x = F.normalize(x)
        x = self.drop(x)

        cores = F.normalize(self.k_embedding.weight)
        items = F.normalize(self.item_embedding.weight)

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode

        probs = None
        mulist = []
        logvarlist = []
        for k in range(self.kfac):
            cates_k = cates[:, k].reshape(1, -1)
            # encoder
            x_k = x * cates_k
            h = self.encoder(x_k)

            mu, logvar = h[:, : self.q_dims[-1]], h[:, self.q_dims[-1] :]
            # mu = F.normalize(mu)

            mulist.append(mu)
            logvarlist.append(logvar)

            z = self.reparameterize(mu, logvar)

            # decoder
            z_k = F.normalize(z)
            logits_k = torch.matmul(z_k, items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = probs_k if (probs is None) else (probs + probs_k)

        logits = torch.log(probs)

        return logits, mulist, logvarlist

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
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight.data)


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

        loss_kld = None
        for i in range(self.model.kfac):
            loss_kld_k = -0.5 * torch.mean(
                torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp(), dim=1)
            )
            # loss_kld_k = -0.5 * torch.mean(torch.sum(1 + logvar[i] - logvar[i].exp(), dim=1))
            loss_kld = loss_kld_k if (loss_kld is None) else (loss_kld + loss_kld_k)

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
    model = MacridVAE([item_num, 600, 200])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    ]

    runner = RecSysRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        engine=engine,
        hparams=hparams,
        loaders=loaders,
        num_epochs=15,
        verbose=True,
        timeit=False,
        callbacks=callbacks,
        logdir="./logs_macridvae",
    )
