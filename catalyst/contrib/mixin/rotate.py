import random
import albumentations as A

from catalyst import utils


class RotateMixin:
    """
    Calculates rotation factor for augmented image
    """
    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "rotation_factor",
        targets_key: str = None,
        rotate_probability: float = 1.,
        hflip_probability: float = 0.5,
        one_hot_classes: int = None
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key
        self.targets_key = targets_key
        self.rotate_probability = rotate_probability
        self.hflip_probability = hflip_probability
        self.rotate = A.RandomRotate90()
        self.hflip = A.HorizontalFlip()
        self.one_hot_classes = one_hot_classes * 8 \
            if one_hot_classes is not None \
            else None

    def __call__(self, dictionary):
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

        if self.targets_key is not None:
            class_rotation_factor = \
                dictionary[self.targets_key] * 8 + rotation_factor
            key = f"class_rotation_{self.targets_key}"
            dictionary[key] = class_rotation_factor

            if self.one_hot_classes is not None:
                one_hot = utils.get_one_hot(
                    class_rotation_factor,
                    self.one_hot_classes
                )
                key = f"class_rotation_{self.targets_key}_one_hot"
                dictionary[key] = one_hot

        return dictionary
