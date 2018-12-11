

class AbstractDataSource:

    @staticmethod
    def prepare_transforms(
            *,
            mode: str,
            stage: str = None,
            **kwargs):
        assert len(kwargs) == 0

    @staticmethod
    def prepare_loaders(
            *,
            stage: str = None,
            n_workers: int = None,
            batch_size: int = None,
            **kwargs):
        assert len(kwargs) == 0
