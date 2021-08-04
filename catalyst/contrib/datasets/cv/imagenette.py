from catalyst.contrib.datasets.cv.misc import ImageClassificationDataset


class Imagenette(ImageClassificationDataset):
    """
    `Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ Dataset.
    """

    name = "imagenette2"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
            "fe2fc210e6bb7c5664d602c3cd71e612",
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
            "e793b78cc4c9e9a4ccc0c1155377a412",
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
            "3df6f0d01a2c9592104656642f5e78a3",
        )
    ]


__all__ = ["Imagenette", "Imagenette160", "Imagenette320"]
