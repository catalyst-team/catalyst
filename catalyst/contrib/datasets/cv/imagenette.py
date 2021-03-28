from catalyst.contrib.datasets.cv.misc import ImageClassificationDataset


class Imagenette(ImageClassificationDataset):
    """
    `Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ Dataset.
    """

    name = "imagenette2"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
            "43b0d8047b7501984c47ae3c08110b62",
        )
    ]


class Imagenette160(ImageClassificationDataset):
    """
    `Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ Dataset
    with images resized so that the shortest size is 160 px.
    """

    name = "imagenette2-160"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
            "0edfc972b5c9817ac36517c0057f3869",
        )
    ]


class Imagenette320(ImageClassificationDataset):
    """
    `Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ Dataset
    with images resized so that the shortest size is 320 px.
    """

    name = "imagenette2-320"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            "3d9f4d75d012a679600ef8ac0c200d28",
        )
    ]


__all__ = ["Imagenette", "Imagenette160", "Imagenette320"]
