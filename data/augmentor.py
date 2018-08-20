
class Augmentor(object):
    def __init__(
            self, dict_key, augment_fn,
            default_kwargs=None):
        self.dict_key = dict_key
        self.augment_fn = augment_fn
        self.default_kwargs = default_kwargs or {}

    def __call__(self, dict_):
        dict_[self.dict_key] = self.augment_fn(
            dict_[self.dict_key], **self.default_kwargs)
        return dict_


class AugmentorKeys(object):
    def __init__(
            self, dict2fn_keys, augment_fn,
            fn_keys=None,
            default_kwargs=None):
        self.dict2fn_keys = dict2fn_keys
        self.fn_keys = fn_keys
        self.augment_fn = augment_fn
        self.default_kwargs = default_kwargs or {}

    def __call__(self, dict_):
        fn_kwargs = {
            value: dict_[key]
            for key, value in self.dict2fn_keys.items()}
        fn_results = self.augment_fn(
            **fn_kwargs, **self.default_kwargs)
        if self.fn_keys is not None:
            fn_results = {
                key: value
                for key, value in zip(self.fn_keys, fn_results)}
        dict_ = {**dict_, **fn_results}
        return dict_
