import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from catalyst import dl


class CustomRunner(dl.Runner):
    """
    Docs.
    """

    def _handle_batch(self, batch):
        """
        Docs.
        """
        state = self.state

        images, _ = batch
        images = images.view(images.size(0), -1)
        bs = images.shape[0]
        z = torch.randn(bs, 128).to(self.device)
        generated_images = self.model["generator"](z)

        # generator step
        # predictions & labels
        generated_labels = torch.ones(bs, 1).to(self.device)
        generated_pred = self.model["discriminator"](generated_images)

        # loss
        loss_generator = F.binary_cross_entropy(
            generated_pred, generated_labels
        )
        state.batch_metrics["loss_generator"] = loss_generator

        # discriminator step
        # real
        images_labels = torch.ones(bs, 1).to(self.device)
        images_pred = self.model["discriminator"](images)
        real_loss = F.binary_cross_entropy(images_pred, images_labels)

        # fake
        generated_labels_ = torch.zeros(bs, 1).to(self.device)
        generated_pred_ = self.model["discriminator"](
            generated_images.detach()
        )
        fake_loss = F.binary_cross_entropy(generated_pred_, generated_labels_)

        # loss
        loss_discriminator = (real_loss + fake_loss) / 2.0
        state.batch_metrics["loss_discriminator"] = loss_discriminator


def main():
    """
    Docs.
    """
    generator = nn.Sequential(nn.Linear(128, 28 * 28), nn.Tanh())
    discriminator = nn.Sequential(nn.Linear(28 * 28, 1), nn.Sigmoid())
    model = nn.ModuleDict(
        {"generator": generator, "discriminator": discriminator}
    )

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0001, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999)
    )
    optimizer = {
        "generator": generator_optimizer,
        "discriminator": discriminator_optimizer,
    }

    loaders = {
        "train": DataLoader(
            MNIST(
                os.getcwd(),
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=32,
        ),
        "valid": DataLoader(
            MNIST(
                os.getcwd(),
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=32,
        ),
    }

    runner = CustomRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=[
            dl.OptimizerCallback(
                optimizer_key="generator", loss_key="loss_generator"
            ),
            dl.OptimizerCallback(
                optimizer_key="discriminator", loss_key="loss_discriminator"
            ),
        ],
        main_metric="loss_generator",
        num_epochs=5,
        logdir="./logs/gan",
        verbose=True,
        check=True,
    )


if __name__ == "__main__":
    no_apex = str(os.environ.get("USE_APEX", "0")) == "0"
    no_ddp = str(os.environ.get("USE_DDP", "0")) == "0"
    if no_apex and no_ddp:
        main()
