from typing import Dict, Callable

class Augmentor(object):
    """
    Augmentation abstraction to use with data dictionaries.
    """
    def __init__(
            self,
            dict_key: str,
            augment_fn: Callable,
            default_kwargs: Dict = None):
        """
        :param dict_key: key to transform
        :param augment_fn: augmentation function to use
        :param default_kwargs: default kwargs for augmentations function
        """
        self.dict_key = dict_key
        self.augment_fn = augment_fn
        self.default_kwargs = default_kwargs or {}

    def __call__(self, dict_):
        dict_[self.dict_key] = self.augment_fn(
            dict_[self.dict_key], **self.default_kwargs)
        return dict_


class AugmentorKeys(object):
    """
    @TODO: ?
    """
    def __init__(
            self,
            dict2fn_keys,
            augment_fn,
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


class AugmentorAlbu:
    '''Augmentor for albumentations interface'''
    def __init__(
            self, dict2fn_keys,  fn_keys, augment_fn):
        '''
        :param dict2fn_keys: list(string) - list of dataset dict_ keys.
               ex: ["image", "target"]
        :param fn_keys: list(string) - list of keys for augment_fn result
               to link dict2fn_keys with augment_fn. ex: ["image", "mask"]
        :param augment_fn: augmentation function
        '''
        assert len(dict2fn_keys) == len(fn_keys), \
            "Parameters dict2fn_keys and fn_keys should have the same length." \
            f"Parameters: {dict2fn_keys, fn_keys}"

        self.dict2fn_keys = dict2fn_keys
        self.fn_keys = fn_keys
        self.augment_fn = augment_fn

    def __call__(self, dict_):
        '''
        :param dict_: dict - item form dataset
        :return dict_: dict - with augmented data
        '''
        # link keys from dict_ with augment_fn keys
        data = {fn_key: dict_[dict_key] for dict_key, fn_key
                in zip(self.dict2fn_keys, self.fn_keys)}

        augmented = self.augment_fn(**data)
        # link keys from augment_fn back to dict_ keys
        results = {dict_key: augmented[fn_key] for dict_key, fn_key
                   in zip(self.dict2fn_keys, self.fn_keys)}

        return {**dict_, **results}
