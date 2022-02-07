# flake8: noqa
from typing import Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl
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


class MultiDAE(nn.Module):
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

        self.encoder = nn.Sequential()
        self.encoder.add_module("normalize", Normalize())
        self.encoder.add_module("dropout", nn.Dropout(dropout))
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            self.encoder.add_module(f"encoder_fc_{i + 1}", nn.Linear(d_in, d_out))
            self.encoder.add_module(f"encoder_tanh_{i + 1}", nn.Tanh())

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
        net = nn.Sequential(self.encoder, self.decoder)
        return net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    set_global_seed(42)

    train_dataset = MovieLens(root=".", train=True, download=True)
    test_dataset = MovieLens(root=".", train=False, download=True)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_train),
        "valid": DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_valid),
    }

    item_num = len(train_dataset[0])
    model = MultiDAE([200, 600, item_num], dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    engine = dl.Engine()
    callbacks = [
        dl.NDCGCallback("logits", "targets", [20, 50, 100]),
        dl.MAPCallback("logits", "targets", [20, 50, 100]),
        dl.MRRCallback("logits", "targets", [20, 50, 100]),
        dl.HitrateCallback("logits", "targets", [20, 50, 100]),
        dl.BackwardCallback("loss"),
        dl.OptimizerCallback("loss", accumulation_steps=1),
    ]

    runner = dl.SupervisedRunner(
        input_key="inputs", output_key="logits", target_key="targets", loss_key="loss"
    )
    runner.train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        engine=engine,
        loaders=loaders,
        num_epochs=100,
        verbose=True,
        timeit=False,
        callbacks=callbacks,
        logdir="./logs_multidae",
    )
