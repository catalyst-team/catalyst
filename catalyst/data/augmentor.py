from typing import Callable, Dict  # isort:skip


class Augmentor:
    """
    Augmentation abstraction to use with data dictionaries.
    """
    def __init__(
        self,
        dict_key: str,
        augment_fn: Callable,
        input_key: str = None,
        output_key: str = None,
        default_kwargs: Dict = None,
    ):
        """
        Args:
            dict_key (str): key to transform
            augment_fn (Callable): augmentation function to use
            input_key (str): ``augment_fn`` input key
            output_key (str): ``augment_fn`` output key
            default_kwargs: default kwargs for augmentations function
        """
        self.dict_key = dict_key
        self.augment_fn = augment_fn
        self.input_key = input_key
        self.output_key = output_key
        self.default_kwargs = default_kwargs or {}

    def __call__(self, dict_: dict):
        """Applies the augmentation"""
        if self.input_key is not None:
            output = self.augment_fn(
                **{self.input_key: dict_[self.dict_key]}, **self.default_kwargs
            )
        else:
            output = self.augment_fn(
                dict_[self.dict_key], **self.default_kwargs
            )

        if self.output_key is not None:
            dict_[self.dict_key] = output[self.output_key]
        else:
            dict_[self.dict_key] = output
        return dict_


class AugmentorKeys:
    """
    Augmentation abstraction to match input and augmentations keys
    """
    def __init__(self, dict2fn_dict: Dict[str, str], augment_fn: Callable):
        """
        Args:
            dict2fn_dict (Dict[str, str]): keys matching dict
                ``{input_key: augment_fn_key}``. For example:
                ``{"image": "image", "mask": "mask"}``
            augment_fn: augmentation function
        """
        self.dict2fn_dict = dict2fn_dict
        self.augment_fn = augment_fn

    def __call__(self, dictionary: dict) -> dict:
        """
        Args:
            dictionary (dict): item from dataset

        Returns:
            dict: dictionaty with augmented data
        """
        # link keys from dict_ with augment_fn keys
        data = {
            fn_key: dictionary[dict_key]
            for dict_key, fn_key in self.dict2fn_dict.items()
        }

        augmented = self.augment_fn(**data)

        # link keys from augment_fn back to dict_ keys
        results = {
            dict_key: augmented[fn_key]
            for dict_key, fn_key in self.dict2fn_dict.items()
        }

        return {**dictionary, **results}


class AugmentorCompose:
    """
    Compose augmentors
    """
    def __init__(self, key2augment_fn: Dict[str, Callable]):
        """
        Args:
            key2augment_fn (Dict[str, Callable]): mapping from input key
                to augmentation function to apply
        """
        self.key2augment_fn = key2augment_fn

    def __call__(self, dictionary: dict) -> dict:
        """
        Args:
            dictionary (dict): item from dataset

        Returns:
            dict: dictionaty with augmented data
        """
        results = {}
        for key, augment_fn in self.key2augment_fn.items():
            results = {**results, **augment_fn({key: dictionary[key]})}

        return {**dictionary, **results}


__all__ = ["Augmentor", "AugmentorKeys"]
