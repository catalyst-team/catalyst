# flake8: noqa
# Needed to collect coverage data
import os
import path

DATA_ROOT = path.Path(os.path.abspath(os.path.dirname(__file__))).abspath()
MOVIELENS_ROOT = (
    path.Path(os.path.abspath(os.path.dirname(__file__)))
    .joinpath("./movielens")
    .abspath()
)
MOVIELENS20M_ROOT = (
    path.Path(os.path.abspath(os.path.dirname(__file__)))
    .joinpath("./movielens20m")
    .abspath()
)


IS_CPU_REQUIRED = os.environ.get("CPU_REQUIRED", "0") == "1"
IS_GPU_REQUIRED = os.environ.get("GPU_REQUIRED", "0") == "1"
IS_GPU_AMP_REQUIRED = os.environ.get("GPU_AMP_REQUIRED", "0") == "1"
IS_DP_REQUIRED = os.environ.get("DP_REQUIRED", "0") == "1"
IS_DP_AMP_REQUIRED = os.environ.get("DP_AMP_REQUIRED", "0") == "1"
IS_DDP_REQUIRED = os.environ.get("DDP_REQUIRED", "0") == "1"
IS_DDP_AMP_REQUIRED = os.environ.get("DDP_AMP_REQUIRED", "0") == "1"
IS_CONFIGS_REQUIRED = os.environ.get("CONFIGS_REQUIRED", "0") == "1"
