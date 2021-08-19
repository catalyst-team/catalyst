from catalyst.contrib.datasets.cv.misc import ImageClassificationDataset


class Imagewoof(ImageClassificationDataset):
    """
    `Imagewoof <https://github.com/fastai/imagenette#imagewoof>`_ Dataset.
    """

    name = "imagewoof2"
    resources = [
        (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz",
            "9aafe18bcdb1632c4249a76c458465ba",
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
            "3d200a7be99704a0d7509be2a9fbfe15",
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
            "0f46d997ec2264e97609196c95897a44",
        )
    ]


__all__ = ["Imagewoof", "Imagewoof160", "Imagewoof320"]
