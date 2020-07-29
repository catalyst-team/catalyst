from catalyst.contrib.datasets.cv.fastai import ImageClassificationDataset


class Imagenette(ImageClassificationDataset):
    name = "imagenette2"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
            "43b0d8047b7501984c47ae3c08110b62",
        )
    ]


class Imagenette160(ImageClassificationDataset):
    name = "imagenette2-160"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
            "0edfc972b5c9817ac36517c0057f3869",
        )
    ]


class Imagenette320(ImageClassificationDataset):
    name = "imagenette2-320"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            "3d9f4d75d012a679600ef8ac0c200d28",
        )
    ]


__all__ = ["Imagenette", "Imagenette160", "Imagenette320"]
