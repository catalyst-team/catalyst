import os

from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst import contrib as ctb, data, dl
import catalyst.contrib.data.transforms as t
from catalyst.contrib.models.simple_conv import SimpleConv
from catalyst.core.callbacks import ControlFlowCallback
import catalyst.data.sampler_inbatch as si
from catalyst.dl.callbacks.metrics.cmc import CMCScoreCallback


def run_ml_pipeline(sampler_inbatch: si.InBatchTripletsSampler) -> float:
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

    dataset_train = ctb.datasets.mnist.MnistMLDataset(
        root=dataset_root, train=True, download=True, transform=transforms,
    )
    sampler = data.sampler.BalanceBatchSampler(
        labels=dataset_train.get_labels(), p=10, k=10
    )
    train_loader = DataLoader(
        dataset=dataset_train, sampler=sampler, batch_size=sampler.batch_size
    )

    dataset_val = ctb.datasets.mnist.MnistQGDataset(
        root=dataset_root, transform=transforms, gallery_fraq=0.2
    )
    val_loader = DataLoader(dataset=dataset_val, batch_size=1024)

    # 2. model and optimizer
    model = SimpleConv(features_dim=16)
    optimizer = Adam(model.parameters(), lr=0.0005)

    # 3. criterion with triplets sampling
    criterion = ctb.nn.criterion.triplet.TripletMarginLossWithSampling(
        margin=0.5, sampler_inbatch=sampler_inbatch
    )

    # 4. training with catalyst Runner
    callbacks = [
        ControlFlowCallback(dl.CriterionCallback(), loaders="train"),
        ControlFlowCallback(
            dl.callbacks.metrics.cmc.CMCScoreCallback(topk_args=[1]),
            loaders="valid",
        ),
        dl.callbacks.PeriodicLoaderCallback(valid=10),
    ]

    runner = dl.SupervisedRunner(device="cuda:0")
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders={"train": train_loader, "valid": val_loader},
        minimize_metric=False,
        verbose=True,
        valid_loader="valid",
        num_epochs=600,
        main_metric="cmc_1",
    )
    return runner.best_valid_metrics["cmc_1"]


def main() -> None:
    """
    This function checks metric learning pipeline with
    different triplets samplers.
    """
    cmc_score_th = 0.97

    all_sampler = si.AllTripletsSampler(max_output_triplets=512)
    hard_sampler = si.HardTripletsSampler(norm_required=False)

    assert run_ml_pipeline(all_sampler) > cmc_score_th
    assert run_ml_pipeline(hard_sampler) > cmc_score_th


if __name__ == "__main__":
    if os.getenv("USE_APEX", "0") == "0" and os.getenv("USE_DDP", "0") == "0":
        main()
