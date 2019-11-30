from catalyst.dl import SupervisedRunner


class BertSupervisedRunner(SupervisedRunner):
    """Wrapper around SupervisedRunner to account for
        input attention masks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         input_key=(
                             "features",
                             "attention_mask",
                         ), **kwargs)


__all__ = ["BertSupervisedRunner"]
