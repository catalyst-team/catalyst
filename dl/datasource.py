

class AbstractDataSource:

    @staticmethod
    def prepare_transforms(*, mode, stage=None):
        raise NotImplementedError

    @staticmethod
    def prepare_loaders(args, data_params, stage=None):
        raise NotImplementedError
