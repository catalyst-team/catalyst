import os

from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import data, dl, utils
from catalyst.contrib import datasets, models, nn
import catalyst.data.cv.transforms.torch as t


def run_ml_pipeline(sampler_inbatch: data.IInbatchTripletSampler) -> float:
    """
    Full metric learning pipeline, including train and val.

    This function is also used as minimal example in README.md, section name:
    'CV - MNIST with Metric Learning'.

    Args:
        sampler_inbatch: sampler to forming triplets

    Returns:
        best metric value
    """
    # 1. train and valid datasets
    dataset_root = "./data"
    transforms = t.Compose([t.ToTensor(), t.Normalize((0.1307,), (0.3081,))])

    dataset_train = datasets.MnistMLDataset(
        root=dataset_root, train=True, download=True, transform=transforms,
    )
    sampler = data.BalanceBatchSampler(
        labels=dataset_train.get_labels(), p=10, k=10
    )
    train_loader = DataLoader(
        dataset=dataset_train, sampler=sampler, batch_size=sampler.batch_size
    )

    dataset_val = datasets.MnistQGDataset(
        root=dataset_root, transform=transforms, gallery_fraq=0.2
    )
    val_loader = DataLoader(dataset=dataset_val, batch_size=1024)

    # 2. model and optimizer
    model = models.SimpleConv(features_dim=16)
    optimizer = Adam(model.parameters(), lr=0.0005)

    # 3. criterion with triplets sampling
    criterion = nn.TripletMarginLossWithSampler(
        margin=0.5, sampler_inbatch=sampler_inbatch
    )

    # 4. training with catalyst Runner
    callbacks = [
        dl.ControlFlowCallback(dl.CriterionCallback(), loaders="train"),
        dl.ControlFlowCallback(
            dl.CMCScoreCallback(topk_args=[1]), loaders="valid"
        ),
        dl.PeriodicLoaderCallback(valid=50),
    ]

    runner = dl.SupervisedRunner(device=utils.get_device())
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders={"train": train_loader, "valid": val_loader},
        minimize_metric=False,
        verbose=True,
        valid_loader="valid",
        num_epochs=50,
        main_metric="cmc01",
    )
    return runner.best_valid_metrics["cmc01"]


def main() -> None:
    """
    This function checks metric learning pipeline with
    different triplets samplers.
    """
    cmc_score_th = 0.9

    # Note! cmc_score should be > 0.97
    # after 600 epoch. Please check it mannually
    # to avoid wasting time of CI pod

    all_sampler = data.AllTripletsSampler(max_output_triplets=512)
    hard_sampler = data.HardTripletsSampler(norm_required=False)
    cluster_sampler = data.HardClusterSampler()

    assert run_ml_pipeline(all_sampler) > cmc_score_th
    assert run_ml_pipeline(hard_sampler) > cmc_score_th
    assert run_ml_pipeline(cluster_sampler) > cmc_score_th


if __name__ == "__main__":
    if os.getenv("USE_APEX", "0") == "0" and os.getenv("USE_DDP", "0") == "0":
        main()
