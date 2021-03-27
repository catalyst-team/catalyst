# flake8: noqa

import os
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.modules import Flatten, GlobalMaxPool2d, Lambda
from catalyst.data.transforms import ToTensor
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def train_experiment(device):
    with TemporaryDirectory() as logdir:
        latent_dim = 128
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
        criterion = {"generator": nn.BCEWithLogitsLoss(), "discriminator": nn.BCEWithLogitsLoss()}
        optimizer = {
            "generator": torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999)),
            "discriminator": torch.optim.Adam(
                discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999)
            ),
        }
        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
            ),
        }

        class CustomRunner(dl.Runner):
            def predict_batch(self, batch):
                batch_size = 1
                # Sample random points in the latent space
                random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)
                # Decode them to fake images
                generated_images = self.model["generator"](random_latent_vectors).detach()
                return generated_images

            def handle_batch(self, batch):
                real_images, _ = batch
                batch_size = real_images.shape[0]

                # Sample random points in the latent space
                random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)

                # Decode them to fake images
                generated_images = self.model["generator"](random_latent_vectors).detach()
                # Combine them with real images
                combined_images = torch.cat([generated_images, real_images])

                # Assemble labels discriminating real from fake images
                labels = torch.cat([torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))]).to(
                    self.device
                )
                # Add random noise to the labels - important trick!
                labels += 0.05 * torch.rand(labels.shape).to(self.device)

                # Discriminator forward
                combined_predictions = self.model["discriminator"](combined_images)

                # Sample random points in the latent space
                random_latent_vectors = torch.randn(batch_size, latent_dim).to(self.device)
                # Assemble labels that say "all real images"
                misleading_labels = torch.zeros((batch_size, 1)).to(self.device)

                # Generator forward
                generated_images = self.model["generator"](random_latent_vectors)
                generated_predictions = self.model["discriminator"](generated_images)

                self.batch = {
                    "combined_predictions": combined_predictions,
                    "labels": labels,
                    "generated_predictions": generated_predictions,
                    "misleading_labels": misleading_labels,
                }

        runner = CustomRunner()
        runner.train(
            engine=dl.DeviceEngine(device),
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            callbacks=[
                dl.CriterionCallback(
                    input_key="combined_predictions",
                    target_key="labels",
                    metric_key="loss_discriminator",
                    criterion_key="discriminator",
                ),
                dl.CriterionCallback(
                    input_key="generated_predictions",
                    target_key="misleading_labels",
                    metric_key="loss_generator",
                    criterion_key="generator",
                ),
                dl.OptimizerCallback(
                    model_key="generator", optimizer_key="generator", metric_key="loss_generator",
                ),
                dl.OptimizerCallback(
                    model_key="discriminator",
                    optimizer_key="discriminator",
                    metric_key="loss_discriminator",
                ),
            ],
            valid_loader="train",
            valid_metric="loss_generator",
            minimize_valid_metric=True,
            num_epochs=1,
            verbose=False,
            logdir=logdir,
        )
        runner.predict_batch(None)[0, 0].cpu().numpy()


def test_finetune_on_cpu():
    train_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_cuda():
    train_experiment("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_finetune_on_cuda_device():
    train_experiment("cuda:1")
