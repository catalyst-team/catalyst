from typing import Dict
import random

import albumentations as albu


class FlareMixin:
    """Calculates flare factor for augmented image."""

    def __init__(
        self,
        input_key: str = "image",
        output_key: str = "flare_factor",
        sunflare_params: Dict = None,
    ):
        """
        Args:
            input_key: input key to use from annotation dict
            output_key: output key to use to store the result
            sunflare_params: params to init
                ``albumentations.RandomSunFlare``
        """
        self.input_key = input_key
        self.output_key = output_key

        self.sunflare_params = sunflare_params or {}
        self.transform = albu.RandomSunFlare(**self.sunflare_params)

    def __call__(self, dictionary):
        """@TODO: Docs. Contribution is welcome."""
        image = dictionary[self.input_key]
        sunflare_factor = 0

        if random.random() < self.transform.p:
            params = self.transform.get_params()
            image = self.transform.apply(image=image, **params)
            sunflare_factor = 1

        dictionary[self.input_key] = image
        dictionary[self.output_key] = sunflare_factor

        return dictionary


__all__ = ["FlareMixin"]
