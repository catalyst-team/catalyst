Multiple input and output keys
==============================================================================

Catalyst supports models with multiple input arguments and multiple outputs.

Suppose that we need to train a siamese network.
Firstly, need to create a dataset that will yield pairs of images and the same class indicator
which later can be used in a contrastive loss.


.. code-block:: python

    import cv2
    import numpy as np
    from torch.utils.data import Dataset

    class SiameseDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            original_image = ... # load image using `idx`
            is_same = np.random.uniform() >= 0.5  # use same or opposite class
            if is_same:
                pair_image = ... # load image from the same class and with index != `idx`
            else:
                pair_image = ... # load image from another class
            label = torch.FloatTensor([is_same])
            return {"first": original_image, "second": pair_image, "labels": label}


Do not forget about contrastive loss:

.. code-block:: python

    import torch.nn as nn

    class ContrastiveLoss(nn.Module):
        def __init__(self, margin=1.0):
            super().__init__()
            self.margin = margin

        def forward(self, l2_distance, labels, **kwargs):
            # ...
            return loss



Suppose you have a model which accepts two tensors - `first` and `second`
and returns embeddings for input batches and distance between them:

.. code-block:: python

    import torch.nn as nn

    class SiameseNet(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_features, in_features * 2),
                nn.ReLU(),
                nn.Linear(in_features * 2, out_features),
            )

        def get_embeddings(self, batch):
            """Generate embeddings for a given batch of images.

            Args:
                batch (torch.Tensor): batch with images,
                    expected shapes - [B, C, H, W].

            Returns:
                embeddings (torch.Tensor) for a given batch of images,
                    output shapes - [B, out_features].
            """
            return self.layers(batch)


        def forward(self, first, second):
            """Forward pass.

            Args:
                first (torch.Tensor): batch with images,
                    expected shapes - [B, C, H, W].
                second (torch.Tensor): batch with images,
                    expected shapes - [B, C, H, W].

            Returns:
                embeddings (torch.Tensor) for a first batch of images,
                    output shapes - [B, out_features]
                embeddings (torch.Tensor) for a second batch of images,
                    output shapes - [B, out_features]
                l2 distance (torch.Tensor) between first and second image embeddings,
                    output shapes - [B,]
            """
            first = self.get_embeddings(first)
            second = self.get_embeddings(second)
            difference = torch.sqrt(torch.sum(torch.pow(first - second, 2), 1))
            return first, second, distance


And then for python API:

.. code-block:: python

    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from catalyst import dl

    dataset = SiameseDataset(...)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    model = SiameseNet(...)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss(margin=1.1)

    runner = dl.SupervisedRunner(
        input_key=["first", "second"],  # model inputs, should be the same as in forward method
        output_key=["first_emb", "second_emb", "l2_distance"],  # model outputs, part of them will be passed to loss
        target_key=["labels"],  # key from dataset
        loss_key="loss",  # key to use for loss values
    )
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=3,
        callbacks=[
            dl.CriterionCallback(
                input_key="l2_distance", target_key="labels", metric_key="loss"
            ),
        ],
        logdir="./siamese_logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )
