<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated DL R&D**

[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Catalyst_Deploy/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Catalyst&tab=projectOverview&guest=1)
[![CodeFactor](https://www.codefactor.io/repository/github/catalyst-team/catalyst/badge)](https://www.codefactor.io/repository/github/catalyst-team/catalyst)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-on%20twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)

</div>

PyTorch framework for Deep Learning research and development.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop. <br/>
Break the cycle - use the Catalyst!

Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md). Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing):
- [Alchemy](https://github.com/catalyst-team/alchemy) - Experiments logging & visualization
- [Catalyst](https://github.com/catalyst-team/catalyst) - Accelerated Deep Learning Research and Development
- [Reaction](https://github.com/catalyst-team/reaction) - Convenient Deep Learning models serving

[Catalyst at AI Landscape](https://landscape.lfai.foundation/selected=catalyst).

----

## Getting started

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import SupervisedRunner

# experiment setup
logdir = "./logdir"
num_epochs = 8

# data
num_samples, num_features = int(1e4), int(1e1)
X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

# model training
runner = SupervisedRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True,
)
```

### Minimal Examples

<details>
<summary>ML - Linear Regression is my profession</summary>
<p>

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import SupervisedRunner

# experiment setup
logdir = "./logdir"
num_epochs = 8

# data
num_samples, num_features = int(1e4), int(1e1)
X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

# model training
runner = SupervisedRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True,
)
```
</p>
</details>

<details>
<summary>CV - MNIST one more time</summary>
<p>

```python
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from catalyst import dl

model = torch.nn.Linear(28 * 28, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=False, download=True,transform=transforms.ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True,transform=transforms.ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):
    def _handle_batch(self, batch):
        x, y = batch
        y_hat = self.model(x.view(x.size(0), -1))
        loss = F.cross_entropy(y_hat, y)
        self.state.batch_metrics["loss"] = loss
        
        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()

runner = CustomRunner()
runner.train(
    model=model, 
    optimizer=optimizer, 
    loaders=loaders, 
    verbose=True,
)
```
</p>
</details>

<details>
<summary>CV - MNIST classification with AutoEncoder</summary>
<p>

```python
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from catalyst import dl


class ClassifyAE(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, hid_features), nn.Tanh())
        self.decoder = nn.Linear(hid_features, in_features)
        self.clf = nn.Linear(hid_features, out_features)
    
    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.clf(z)
        x_ = self.decoder(z)
        return y_hat, x_


model = ClassifyAE(28 * 28, 128, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=False, download=True,transform=transforms.ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True,transform=transforms.ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):
    def _handle_batch(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat, x_ = self.model(x)
        loss_clf = F.cross_entropy(y_hat, y)
        loss_ae = F.mse_loss(x_, x)
        loss = loss_clf + loss_ae
        
        self.state.batch_metrics = {
            "loss_clf": loss_clf,
            "loss_ae": loss_ae,
            "loss": loss
        }
        
        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()

runner = CustomRunner()
runner.train(
    model=model, 
    optimizer=optimizer, 
    loaders=loaders, 
    verbose=True,
)
```
</p>
</details>

<details>
<summary>GAN - MNIST, flatten version</summary>
<p>

```python
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from catalyst import dl


generator = nn.Sequential(nn.Linear(128, 28 * 28), nn.Tanh())
discriminator = nn.Sequential(nn.Linear(28 * 28, 1), nn.Sigmoid())
model = nn.ModuleDict({"generator": generator, "discriminator": discriminator})

generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer = {
    "generator": generator_optimizer,
    "discriminator": discriminator_optimizer,
}

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=False, download=True,transform=transforms.ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True,transform=transforms.ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):
    
    def _handle_batch(self, batch):
        state = self.state
        
        images, _ = batch
        images = images.view(images.size(0), -1)
        bs = images.shape[0]
        z = torch.randn(bs, 128).to(self.device)
        generated_images = self.model["generator"](z)
        
        # generator step
        ## predictions & labels
        generated_labels = torch.ones(bs, 1).to(self.device)
        generated_pred = self.model["discriminator"](generated_images)

        ## loss
        loss_generator = F.binary_cross_entropy(generated_pred, generated_labels)
        state.batch_metrics["loss_generator"] = loss_generator

        # discriminator step
        ## real
        images_labels = torch.ones(bs, 1).to(self.device)
        images_pred = self.model["discriminator"](images)
        real_loss = F.binary_cross_entropy(images_pred, images_labels)

        ## fake
        generated_labels_ = torch.zeros(bs, 1).to(self.device)
        generated_pred_ = self.model["discriminator"](generated_images.detach())
        fake_loss = F.binary_cross_entropy(generated_pred_, generated_labels_)

        ## loss
        loss_discriminator = (real_loss + fake_loss) / 2.0
        state.batch_metrics["loss_discriminator"] = loss_discriminator

runner = CustomRunner()
runner.train(
    model=model, 
    optimizer=optimizer,
    loaders=loaders,
    callbacks=[
        dl.OptimizerCallback(
            optimizer_key="generator", 
            loss_key="loss_generator"
        ),
        dl.OptimizerCallback(
            optimizer_key="discriminator", 
            loss_key="loss_discriminator"
        ),
    ],
    main_metric="loss_generator",
    num_epochs=5,
    logdir="./logs/gan",
    verbose=True,
)
```
</p>
</details>

[Demo with minimal examples for ML, CV, NLP, GANs and RecSys](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/demo.ipynb)

For Catalyst.RL introduction, please follow [Catalyst.RL repo](https://github.com/catalyst-team/catalyst-rl).


## Table of Contents
- [Overview](#overview)
  * [Installation](#installation)
  * [Features](#features)
  * [Structure](#structure)
  * [Tests](#tests)
- [Catalyst](#catalyst)
  * [Tutorials](#tutorials)
  * [Projects](#projects)
  * [Tools and pipelines](#tools-and-pipelines)
  * [Talks and videos](#talks-and-videos)
- [Community](#community)
  * [Contribution guide](#contribution-guide)
  * [User feedback](#user-feedback)
  * [Trusted by](#trusted-by)
  * [Supported by](#supported-by)
  * [Citation](#citation)


## Overview
Catalyst helps you write compact
but full-featured Deep Learning pipelines in a few lines of code.
You get a training loop with metrics, early-stopping, model checkpointing
and other features without the boilerplate.

### Installation

Common installation:
```bash
pip install -U catalyst
```

<details>
<summary>Specific versions with additional requirements</summary>
<p>

```bash
pip install catalyst[ml]         # installs DL+ML based catalyst
pip install catalyst[cv]         # installs DL+CV based catalyst
pip install catalyst[nlp]        # installs DL+NLP based catalyst
pip install catalyst[ecosystem]  # installs Catalyst.Ecosystem
pip install catalyst[contrib]    # installs DL+contrib based catalyst
pip install catalyst[all]        # installs everything
# and master version installation
pip install git+https://github.com/catalyst-team/catalyst@master --upgrade
```
</p>
</details>

Catalyst is compatible with: Python 3.6+. PyTorch 1.0.0+.

### Features
- Universal train/inference loop.
- Configuration files for model/data hyperparameters.
- Reproducibility – all source code and environment variables will be saved.
- Callbacks – reusable train/inference pipeline parts with easy customization.
- Training stages support.
- Deep Learning best practices - SWA, AdamW, Ranger optimizer, OneCycle, and more.
- Developments best practices - fp16 support, distributed training, slurm.


### Structure
- **contrib** - additional modules contributed by Catalyst users.
- **core** - framework core with main abstractions - 
    Experiment, Runner, Callback and State.
- **data** - useful tools and scripts for data processing.
- **DL** – runner for training and inference,
   all of the classic ML and CV/NLP/RecSys metrics
   and a variety of callbacks for training, validation
   and inference of neural networks.
- **utils** - typical utils for Deep Learning research.


### Tests
All the Catalyst code is [tested rigorously with every new PR](./tests).

In fact, we train a number of different models for various of tasks - 
image classification, image segmentation, text classification, GAN training.
During the tests, we compare their convergence metrics in order to verify 
the correctness of the training procedure and its reproducibility.

Overall, Catalyst guarantees fully tested, correct and reproducible 
best practices for the automated parts.

## Catalyst

### Tutorials
- [Demo with minimal examples](./examples/notebooks/demo.ipynb) for ML, CV, NLP, GANs and RecSys [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/demo.ipynb)
- Detailed [classification tutorial](./examples/notebooks/classification-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb)
- Advanced [segmentation tutorial](./examples/notebooks/segmentation-tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/segmentation-tutorial.ipynb)
- Comprehensive [classification pipeline](https://github.com/catalyst-team/classification)
- Binary and semantic [segmentation pipeline](https://github.com/catalyst-team/segmentation)
- [Beyond fashion: Deep Learning with Catalyst (Config API)](https://evilmartians.com/chronicles/beyond-fashion-deep-learning-with-catalyst)
- [Tutorial from Notebook API to Config API (RU)](https://github.com/Bekovmi/Segmentation_tutorial)

API documentation and an overview of the library can be found here
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html). <br/>
In the **[examples folder](examples)**
of the repository, you can find advanced tutorials and Catalyst best practices.


### Projects
- [Kaggle Quick, Draw! Doodle Recognition Challenge](https://github.com/ngxbac/Kaggle-QuickDraw) - 11th place solution
- [Catalyst.RL - NeurIPS 2018: AI for Prosthetics Challenge](https://github.com/Scitator/neurips-18-prosthetics-challenge) – 3rd place solution
- [CamVid Segmentation Example](https://github.com/BloodAxe/Catalyst-CamVid-Segmentation-Example) - Example of semantic segmentation for CamVid dataset
- [Notebook API tutorial for segmentation in Understanding Clouds from Satellite Images Competition](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools/)
- [Kaggle Google Landmark 2019](https://github.com/ngxbac/Kaggle-Google-Landmark-2019) - 30th place solution
- [Hierarchical attention for sentiment classification with visualization](https://github.com/neuromation/ml-recipe-hier-attention)
- [Pediatric bone age assessment](https://github.com/neuromation/ml-recipe-bone-age)
- [iMet Collection 2019 - FGVC6](https://github.com/ngxbac/Kaggle-iMet) - 24th place solution
- [ID R&D Anti-spoofing Challenge](https://github.com/bagxi/idrnd-anti-spoofing-challenge-solution) - 14th place solution
- [Implementation of paper "Tell Me Where to Look: Guided Attention Inference Network"](https://github.com/ngxbac/GAIN)
- [Catalyst.RL - NeurIPS 2019: Learn to Move - Walk Around](https://github.com/Scitator/learning-to-move-starter-kit) – starter kit
- [Catalyst.RL - NeurIPS 2019: Animal-AI Olympics](https://github.com/Scitator/animal-olympics-starter-kit) - starter kit
- [NeurIPS 2019: Recursion Cellular Image Classification](https://github.com/ngxbac/Kaggle-Recursion-Cellular) - 4th place solution
- [MICCAI 2019: Automatic Structure Segmentation for Radiotherapy Planning Challenge 2019](https://github.com/ngxbac/StructSeg2019) 
  * 3rd place solution for `Task 3: Organ-at-risk segmentation from chest CT scans`
  * and 4th place solution for `Task 4: Gross Target Volume segmentation of lung cancer`
- [Kaggle Seversteal steel detection](https://github.com/bamps53/kaggle-severstal) - 5th place solution
- [Implementation of paper "Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks"](https://github.com/yukkyo/PyTorch-FilterResponseNormalizationLayer)
- [RSNA Intracranial Hemorrhage Detection](https://github.com/ngxbac/Kaggle-RSNA) - 5th place solution
- [Implementation of paper "Utterance-level Aggregation For Speaker Recognition In The Wild"](https://github.com/ptJexio/Speaker-Recognition)
- [APTOS 2019 Blindness Detection](https://github.com/BloodAxe/Kaggle-2019-Blindness-Detection) – 7th place solution
- [Catalyst.RL - NeurIPS 2019: Learn to Move - Walk Around](https://github.com/Scitator/run-skeleton-run-in-3d) – 2nd place solution
- [xView2 Damage Assessment Challenge](https://github.com/BloodAxe/xView2-Solution) - 3rd place solution
- [Inria Segmentation Example](https://github.com/BloodAxe/Catalyst-Inria-Segmentation-Example) - An example of training segmentation model for Inria Sattelite Segmentation Challenge
- [iglovikov_segmentation](https://github.com/ternaus/iglovikov_segmentation) - Semantic segmentation pipeline using Catalyst


### Tools and pipelines
- [Catalyst.RL](https://github.com/Scitator/catalyst-rl-framework) – A Distributed Framework for Reproducible RL Research by [Scitator](https://github.com/Scitator)
- [Catalyst.Classification](https://github.com/catalyst-team/classification) - Comprehensive classification pipeline with Pseudo-Labeling by [Bagxi](https://github.com/bagxi) and [Pdanilov](https://github.com/pdanilov)
- [Catalyst.Segmentation](https://github.com/catalyst-team/segmentation) - Segmentation pipelines - binary, semantic and instance, by [Bagxi](https://github.com/bagxi)
- [Catalyst.Detection](https://github.com/catalyst-team/detection) - Anchor-free detection pipeline by [Avi2011class](https://github.com/Avi2011class) and [TezRomacH](https://github.com/TezRomacH)
- [Catalyst.GAN](https://github.com/catalyst-team/gan) - Reproducible GANs pipelines by [Asmekal](https://github.com/asmekal)
- [Catalyst.Neuro](https://github.com/catalyst-team/neuro) - Brain image analysis project, in collaboration with [TReNDS Center](https://trendscenter.org)
- [MLComp](https://github.com/catalyst-team/mlcomp) – distributed DAG framework for machine learning with UI by [Lightforever](https://github.com/lightforever)
- [Pytorch toolbelt](https://github.com/BloodAxe/pytorch-toolbelt) - PyTorch extensions for fast R&D prototyping and Kaggle farming by [BloodAxe](https://github.com/BloodAxe)
- [Helper functions](https://github.com/ternaus/iglovikov_helper_functions) - An unstructured set of helper functions by [Ternaus](https://github.com/ternaus)


### Talks and videos
- [Catalyst-team YouTube channel](https://www.youtube.com/channel/UC39Z1Cwr9n8DVpuXcsyi9FQ)
- [Catalyst.RL – reproducible RL research framework](https://docs.google.com/presentation/d/1U6VWIwQnQDGtu6a1x61tt3AlxCJ1-A1EYKd8lR9tKos/edit?usp=sharing) at [Stachka](https://nastachku.ru/archive/2019_innopolis/index.php?dispatch=products.view&product_id=3650)
- [Catalyst.DL – reproducible DL research framework (rus)](https://youtu.be/EfG8iwFNdWg) and [slides (eng)](https://docs.google.com/presentation/d/1TL7N_H31zDFShVbKzLfMC3DYw4e1psj6ScDN8spKQlk/edit?usp=sharing) at [RIF](http://rifvrn.ru/program/catalyst-dl-fast-reproducible-dl-4-html)
- [Catalyst.DL – reproducible DL research framework (rus)](https://youtu.be/7xyMP_5eA8c?t=8964) and [slides (eng)](https://docs.google.com/presentation/d/1XGubfTWvpiJrMyKNx2G6GtAq68y2__sDmx30eSdSRZs/edit?usp=sharing) at [AI-Journey](https://ai-journey.ru/conference-moscow/broadcast?page=2&per-page=12)
- [Catalyst.DL – fast & reproducible DL](https://docs.google.com/presentation/d/1fbF4PMl092kIdjJTw3olR3wI2cl_P2ttN3c9-WTh1gA/edit?usp=sharing) at [Datastart](https://datastart.ru/msk-autumn-2019/)
- [Catalyst.RL - NeurIPS 2019: Learn to Move - Walk Around](https://www.youtube.com/watch?v=PprDcJHrFdg&feature=youtu.be&t=4020) and [slides (eng)](https://docs.google.com/presentation/d/1g4g_Rxp9M3xAHwpp_hNzC87L9Gvum3H09g2DIQn1Taw/edit?usp=sharing) at RL reading group Meetup
- [Catalyst – accelerated DL & RL (rus)](https://youtu.be/Rmo2rx5V3v8?t=77) and [slides (eng)](https://docs.google.com/presentation/d/1xMZMjSwJfM5mZMK7pHp6hVI0FxPyZOpRtBZ0J2l1AaY/edit?fbclid=IwAR1q4XJVqYdD-a5oO2n68Y4xHvChIeOSjCSmlUYqrjIzneYpehzF8PiNdMc#slide=id.g75815b5293_0_202) at [Facebook Developer Circle: Moscow | ML & AI Meetup](https://www.facebook.com/groups/475428499888062/)
- [Catalyst.RL - Learn to Move - Walk Around 2nd place solution](https://docs.google.com/presentation/d/14UzYAURBulLjuCbQRnNeROhZ74h51-o460DPTkKMrwo/edit?usp=sharing) at NeurIPS competition track
- [Open Source ML 2019 edition](https://docs.google.com/presentation/d/1A-kwek7USA-j2Nn4n8PmLUQ1PdeUzkkViwXST7RyL-w/edit?usp=sharing) at [Datafest.elka](https://datafest.ru/elka/)


## Community

### Contribution guide

We appreciate all contributions.
If you are planning to contribute back bug-fixes,
please do so without any further discussion.
If you plan to contribute new features, utility functions or extensions,
please first open an issue and discuss the feature with us.

- Please see the [contribution guide](CONTRIBUTING.md) for more information.
- By participating in this project, you agree to abide by its [Code of Conduct](CODE_OF_CONDUCT.md).


### User feedback

We have created `catalyst.team.core@gmail.com` for "user feedback".
- If you like the project and want to say thanks, this the right place.
- If you would like to start a collaboration between your team and Catalyst team to do better Deep Learning R&D - you are always welcome.
- If you just don't like Github issues and this ways suits you better - feel free to email us.
- Finally, if you do not like something, please, share it with us and we can see how to improve it.

We appreciate any type of feedback. Thank you!


### Trusted by
- [Awecom](https://www.awecom.com)
- Researchers@[Center for Translational Research in Neuroimaging and Data Science (TReNDS)](https://trendscenter.org)
- Researchers@[Emory University](https://www.emory.edu)
- [Evil Martians](https://evilmartians.com)
- Researchers@[Georgia Institute of Technology](https://www.gatech.edu)
- Researchers@[Georgia State University](https://www.gsu.edu)
- [Helios](http://helios.to)
- [HPCD Lab](https://www.hpcdlab.com)
- [iFarm](https://ifarmproject.com)
- [Kinoplan](http://kinoplan.io/)
- Researchers@[Moscow Institute of Physics and Technology](https://mipt.ru/english/)
- [Neuromation](https://neuromation.io)
- [Poteha Labs](https://potehalabs.com/en/)
- [Provectus](https://provectus.com)
- Researchers@[Skolkovo Institute of Science and Technology](https://www.skoltech.ru/en)
- [SoftConstruct](https://www.softconstruct.io/)
- Researchers@[Tinkoff](https://www.tinkoff.ru/eng/)
- Researchers@[Yandex.Research](https://research.yandex.com)


### Supported by
- [HostKey](https://www.hostkey.com)
- [Moscow Institute of Physics and Technology](https://mipt.ru/english/)


### Citation

Please use this bibtex if you want to cite this repository in your publications:

    @misc{catalyst,
        author = {Kolesnikov, Sergey},
        title = {Accelerated DL R&D},
        year = {2018},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/catalyst-team/catalyst}},
    }
