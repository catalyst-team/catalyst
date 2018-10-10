

class AbstractDataSource:

    @staticmethod
    def prepare_transforms(*, mode, stage=None, **kwargs):
        assert len(kwargs) == 0

    @staticmethod
    def prepare_loaders(*, args, stage=None, **kwargs):
        assert len(kwargs) == 0
