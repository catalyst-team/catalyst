from catalyst.utils.data.mixup import compute_mixup_lambda, mixup


class MixupAugmentor:

    def __init__(self, alpha: float, share_lambda: bool = True):
        self.alpha = alpha
        self.share_lambda = share_lambda

    def __call__(self, dict_):
        bs = len(dict_[list(dict_.keys())[0]])
        lambda_ = compute_mixup_lambda(bs, self.alpha, self.share_lambda)

        dict_ = {
            key: mixup(value, lambda_=lambda_)
            for key, value in dict_.items()
        }

        return dict_
