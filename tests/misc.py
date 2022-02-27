from pathlib import Path
import subprocess

import torch
from torch.utils import data


class TensorDataset(data.TensorDataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *args: tensors that have the same size of the first dimension.
    """

    def __init__(self, *args: torch.Tensor, **kwargs) -> None:
        super().__init__(*args)


def run_experiment_from_configs(
    config_dir: Path, main_config: str, *auxiliary_configs: str
) -> None:
    """Runs experiment from config (for Config API tests)."""
    main_config = str(config_dir / main_config)
    auxiliary_configs = " ".join(str(config_dir / c) for c in auxiliary_configs)

    script = Path("catalyst", "contrib", "scripts", "run.py")
    cmd = f"python {script} -C {main_config} {auxiliary_configs}"
    subprocess.run(cmd.split(), check=True)
