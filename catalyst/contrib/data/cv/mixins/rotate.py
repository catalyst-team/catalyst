import random

import albumentations as albu

from catalyst import utils


class RotateMixin:
    """Calculates rotation factor for augmented image."""

    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "rotation_factor",
        target_key: str = None,
        rotate_probability: float = 1.0,
        hflip_probability: float = 0.5,
        one_hot_classes: int = None,
    ):
        """
        Args:
            input_key: input key to use from annotation dict
            output_key: output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key
        self.target_key = target_key
        self.rotate_probability = rotate_probability
        self.hflip_probability = hflip_probability
        self.rotate = albu.RandomRotate90()
        self.hflip = albu.HorizontalFlip()
        self.one_hot_classes = one_hot_classes * 8 if one_hot_classes is not None else None

    def __call__(self, dictionary):
        """@TODO: Docs. Contribution is welcome."""
        image = dictionary[self.input_key]
        rotation_factor = 0

        if random.random() < self.rotate_probability:
            rotation_factor = self.rotate.get_params()["factor"]
            image = self.rotate.apply(img=image, factor=rotation_factor)

        if random.random() < self.hflip_probability:
            rotation_factor += 4
            image = self.hflip.apply(img=image)

        dictionary[self.input_key] = image
        dictionary[self.output_key] = rotation_factor

        if self.target_key is not None:
            class_rotation_factor = dictionary[self.target_key] * 8 + rotation_factor
            key = f"class_rotation_{self.target_key}"
            dictionary[key] = class_rotation_factor

            if self.one_hot_classes is not None:
                one_hot = utils.get_one_hot(class_rotation_factor, self.one_hot_classes)
                key = f"class_rotation_{self.target_key}_one_hot"
                dictionary[key] = one_hot

        return dictionary


__all__ = ["RotateMixin"]
