from catalyst.contrib.datasets.cv.misc import ImageClassificationDataset


class Imagewoof(ImageClassificationDataset):
    """
    `Imagewoof <https://github.com/fastai/imagenette#imagewoof>`_ Dataset.
    """

    name = "imagewoof2"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz",
            "5eaf5bbf4bf16a77c616dc6e8dd5f8e9",
        )
    ]


class Imagewoof160(ImageClassificationDataset):
    """
    `Imagewoof <https://github.com/fastai/imagenette#imagewoof>`_ Dataset
    with images resized so that the shortest size is 160 px.
    """

    name = "imagewoof2-160"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz",
            "fcd23cc7dfce8837c95a8f9d63a128b7",
        )
    ]


class Imagewoof320(ImageClassificationDataset):
    """
    `Imagewoof <https://github.com/fastai/imagenette#imagewoof>`_ Dataset
    with images resized so that the shortest size is 320 px.
    """

    name = "imagewoof2-320"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
            "af65be7963816efa949fa3c3b4947740",
        )
    ]


__all__ = ["Imagewoof", "Imagewoof160", "Imagewoof320"]
