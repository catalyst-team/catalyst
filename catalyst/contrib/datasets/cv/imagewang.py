from catalyst.contrib.datasets.cv.misc import ImageClassificationDataset


class Imagewang(ImageClassificationDataset):
    """
    `Imagewang <https://github.com/fastai/imagenette#image%E7%BD%91>`_ Dataset.
    """

    name = "imagewang"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewang.tgz",
            "46f9749616a29837e7cd67b103396f6e",
        )
    ]


class Imagewang160(ImageClassificationDataset):
    """
    `Imagewang <https://github.com/fastai/imagenette#image%E7%BD%91>`_ Dataset
    with images resized so that the shortest size is 160 px.
    """

    name = "imagewang-160"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewang-160.tgz",
            "1dc388d37d1dc52836c06749e14e37bc",
        )
    ]


class Imagewang320(ImageClassificationDataset):
    """
    `Imagewang <https://github.com/fastai/imagenette#image%E7%BD%91>`_ Dataset
    with images resized so that the shortest size is 320 px.
    """

    name = "imagewang-320"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewang-320.tgz",
            "ff01d7c126230afce776bdf72bda87e6",
        )
    ]


__all__ = ["Imagewang", "Imagewang160", "Imagewang320"]
