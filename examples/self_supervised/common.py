from datasets import datasets


def add_arguments(parser) -> None:
    """Function to add common arguments to argparse:
    feature_dim: Feature dim for latent vector
    temperature: Temperature used in softmax
    batch_size: Number of images in each mini-batch
    epochs: Number of sweeps over the dataset to train
    num_workers: Number of workers to process a dataloader
    logdir: Logs directory (tensorboard, weights, etc)
    dataset: Dataset: CIFAR-10, CIFAR-100 or STL10

    Args:
        parser: argparser like object
    """
    parser.add_argument(
        "--feature_dim",
        default=128,
        type=int,
        help="Feature dim for latent vector",
    )
    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="Temperature used in softmax",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=float,
        help="Number of workers to process a dataloader",
    )
    parser.add_argument(
        "--logdir",
        default="./logdir",
        type=str,
        help="Logs directory (tensorboard, weights, etc)",
    )
    parser.add_argument(
        "--dataset",
        default="CIFAR-10",
        type=str,
        choices=datasets.keys(),
        help="Dataset: CIFAR-10, CIFAR-100 or STL10",
    )
