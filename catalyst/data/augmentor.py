from typing import Dict, Callable


class Augmentor:
    """
    Augmentation abstraction to use with data dictionaries.
    """

    def __init__(
        self, dict_key: str, augment_fn: Callable, default_kwargs: Dict = None
    ):
        """
        Args:
            dict_key: key to transform
            augment_fn: augmentation function to use
            default_kwargs: default kwargs for augmentations function
        """
        self.dict_key = dict_key
        self.augment_fn = augment_fn
        self.default_kwargs = default_kwargs or {}

    def __call__(self, dict_):
        dict_[self.dict_key
              ] = self.augment_fn(dict_[self.dict_key], **self.default_kwargs)
        return dict_


class AugmentorKeys:
    """ Augmentation abstraction to match input and augmentations keys"""

    def __init__(self, dict2fn_dict: Dict[str, str], augment_fn: Callable):
        """
        :param dict2fn_dict: keys matching dict {input_key: augment_fn_key}
               ex: {"image": "image", "mask": "mask"}
        :param augment_fn: augmentation function
        """

        self.dict2fn_dict = dict2fn_dict
        self.augment_fn = augment_fn

    def __call__(self, dict_):
        """
        :param dict_: dict - item form dataset
        :return dict_: dict - with augmented data
        """
        # link keys from dict_ with augment_fn keys
        data = {
            fn_key: dict_[dict_key]
            for dict_key, fn_key in self.dict2fn_dict.items()
        }

        augmented = self.augment_fn(**data)
        # link keys from augment_fn back to dict_ keys
        results = {
            dict_key: augmented[fn_key]
            for dict_key, fn_key in self.dict2fn_dict.items()
        }

        return {**dict_, **results}


__all__ = ["Augmentor", "AugmentorKeys"]
