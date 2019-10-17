from catalyst.dl import SupervisedRunner


class BertSupervisedRunner(SupervisedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, input_key=(
            "features",
            "mask",
        ), **kwargs)


__all__ = ["BertSupervisedRunner"]
