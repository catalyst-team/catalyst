

class AbstractDataSource:

    @staticmethod
    def prepare_transforms(*, mode):
        raise NotImplementedError

    @staticmethod
    def prepare_loaders(args, data_params):
        raise NotImplementedError

