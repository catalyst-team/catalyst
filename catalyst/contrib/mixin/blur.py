from typing import List
import random
import numpy as np

import albumentations as A


class BlurMixin:
    """
    Calculates blur factor for augmented image
    """
    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "blur_factor",
        blur_min: int = 3,
        blur_max: int = 9,
        blur: List[str] = None
    ):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key

        self.blur_min = blur_min
        self.blur_max = blur_max
        blur = blur or ["Blur"]
        self.blur = [A.__dict__[x]() for x in blur]
        self.num_blur = len(self.blur)
        self.num_blur_classes = blur_max - blur_min + 1 + 1
        self.blur_probability = \
            (self.num_blur_classes - 1) / self.num_blur_classes

    def __call__(self, dictionary):
        image = dictionary[self.input_key]
        blur_factor = 0

        if random.random() < self.blur_probability:
            blur_fn = np.random.choice(self.blur)
            blur_factor = int(
                np.random.randint(self.blur_min, self.blur_max) -
                self.blur_min + 1
            )
            image = blur_fn.apply(image=image, ksize=blur_factor)

        dictionary[self.input_key] = image
        dictionary[self.output_key] = blur_factor

        return dictionary
