# flake8: noqa
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.modules import Flatten, GlobalMaxPool2d, Lambda
from catalyst.data.cv import ToTensor

LATENT_DIM = 128


class CustomRunner(dl.Runner):
    def _handle_batch(self, batch):
        real_images, _ = batch
        batch_metrics = {}

        # Sample random points in the latent space
        batch_size = real_images.shape[0]
        random_latent_vectors = torch.randn(batch_size, LATENT_DIM).to(
            self.device
        )

        # Decode them to fake images
        generated_images = self.model["generator"](
            random_latent_vectors
        ).detach()
        # Combine them with real images
        combined_images = torch.cat([generated_images, real_images])

        # Assemble labels discriminating real from fake images
        labels = torch.cat(
            [torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))]
        ).to(self.device)
        # Add random noise to the labels - important trick!
        labels += 0.05 * torch.rand(labels.shape).to(self.device)

        # Train the discriminator
        predictions = self.model["discriminator"](combined_images)
        batch_metrics[
            "loss_discriminator"
        ] = F.binary_cross_entropy_with_logits(predictions, labels)

        # Sample random points in the latent space
        random_latent_vectors = torch.randn(batch_size, LATENT_DIM).to(
            self.device
        )
        # Assemble labels that say "all real images"
        misleading_labels = torch.zeros((batch_size, 1)).to(self.device)

        # Train the generator
        generated_images = self.model["generator"](random_latent_vectors)
        predictions = self.model["discriminator"](generated_images)
        batch_metrics["loss_generator"] = F.binary_cross_entropy_with_logits(
            predictions, misleading_labels
        )

        self.batch_metrics.update(**batch_metrics)


def main():
    generator = nn.Sequential(
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        nn.Linear(128, 128 * 7 * 7),
        nn.LeakyReLU(0.2, inplace=True),
        Lambda(lambda x: x.view(x.size(0), 128, 7, 7)),
        nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2), padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2), padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 1, (7, 7), padding=3),
        nn.Sigmoid(),
    )
    discriminator = nn.Sequential(
        nn.Conv2d(1, 64, (3, 3), stride=(2, 2), padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        GlobalMaxPool2d(),
        Flatten(),
        nn.Linear(128, 1),
    )

    model = {"generator": generator, "discriminator": discriminator}
    optimizer = {
        "generator": torch.optim.Adam(
            generator.parameters(), lr=0.0003, betas=(0.5, 0.999)
        ),
        "discriminator": torch.optim.Adam(
            discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999)
        ),
    }
    loaders = {
        "train": DataLoader(
            MNIST("./data", train=True, download=True, transform=ToTensor(),),
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
                optimizer_key="generator", metric_key="loss_generator"
            ),
            dl.OptimizerCallback(
                optimizer_key="discriminator", metric_key="loss_discriminator"
            ),
        ],
        main_metric="loss_generator",
        num_epochs=20,
        verbose=True,
        logdir="./logs_gan",
        check=True,
    )


if __name__ == "__main__":
    if os.getenv("USE_APEX", "0") == "0" and os.getenv("USE_DDP", "0") == "0":
        main()
