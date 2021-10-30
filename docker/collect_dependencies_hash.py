import hashlib
import os

sha = hashlib.sha1(
    "".join(
        [
            open(x).read() if y else ""
            for x, y in [
                ("docker/Dockerfile", True),
                ("requirements/requirements.txt", True),
                ("requirements/requirements-dev.txt", os.environ.get("CATALYST_DEV", "0") == "1"),
                ("requirements/requirements-cv.txt", os.environ.get("CATALYST_CV", "0") == "1"),
                ("requirements/requirements-ml.txt", os.environ.get("CATALYST_ML", "0") == "1"),
                (
                    "requirements/requirements-hydra.txt",
                    os.environ.get("CATALYST_HYDRA", "0") == "1",
                ),
                (
                    "requirements/requirements-optuna.txt",
                    os.environ.get("CATALYST_OPTUNA", "0") == "1",
                ),
                (
                    "requirements/requirements-onnx.txt",
                    os.environ.get("CATALYST_ONNX", "0") == "1",
                ),
                (
                    "requirements/requirements-onnx-gpu.txt",
                    os.environ.get("CATALYST_ONNX_GPU", "0") == "1",
                ),
            ]
        ]
    ).encode()
)
print(sha.hexdigest())
