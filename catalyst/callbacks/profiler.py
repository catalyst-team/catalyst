from catalyst.core.callback import Callback
from catalyst.core.runner import IRunner


class ProfilerCallback(Callback):
    """
    Performs the profiler step for the PyTorch:1.8 profiler
    Args:
        profiler (torch.profiler.profile): The instantiated PyTorch profiler object
    
    Example:
        .. code-block:: python
            import torch  # noqa
            from torch import nn, optim  # noqa
            from torch.utils.data import DataLoader  # noqa
            from catalyst import dl  # noqa
            from catalyst.data.transforms import ToTensor  # noqa
            from catalyst.contrib.datasets import MNIST  # noqa
            from catalyst.callbacks.batch_overfit import BatchOverfitCallback  # noqa
            from catalyst.callbacks.misc import TimerCallback  # noqa
            from catalyst.utils.misc import set_global_seed  # noqa
            from catalyst.callbacks.profiler import ProfilerCallback  # noqa
            profiler = torch.profiler.profile(
                profile_memory=True,
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2))
            with profiler:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                set_global_seed(42)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.02)
                loaders = {
                    "train": DataLoader(
                        MNIST('./', train=True, download=True, transform=ToTensor()),
                        batch_size=256, num_workers=0, pin_memory=True),
                    "valid": DataLoader(
                        MNIST('./', train=False, download=True, transform=ToTensor()),
                        batch_size=256, num_workers=0, pin_memory=True),
                }
                runner = dl.SupervisedRunner()
                # model training
                runner.train(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    loaders=loaders,
                    num_epochs=1,
                    logdir="./logs",
                    valid_loader="valid",
                    valid_metric="loss",
                    minimize_valid_metric=True,
                    verbose=True,
                    callbacks=[ProfilerCallback(profiler)])
                print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                print(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                if torch.cuda.is_available():
                    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    """
    
    
    def __init__(self, profiler=None):
        """
        Performs the profiler step for the PyTorch:1.8 profiler
        Args:
            profiler (torch.profiler.profile): The instantiated PyTorch profiler object
        """
        # Order = 0 means that this has the highest execution priority
        super().__init__(
            order=0, node=1
        )  # Node = 1 means that the callback is only executed on the Main thread.
        self._profiler = profiler

    def on_batch_end(self, runner: IRunner):
        """
        On batch end action. The profiler performs a step for each batch
        Args:
            runner: runner for the experiment.
        """
        if self._profiler is not None:
            self._profiler.step()
