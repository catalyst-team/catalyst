from typing import Any, Callable, Dict

from torch.utils.data import Dataset


class SelfSupervisedDatasetWrapper(Dataset):
    """The Self Supervised Dataset.

    The class implemets contrastive logic (see Figure 2 from `A Simple Framework
    for Contrastive Learning of Visual Representations`_.)

    Args:
        dataset: original dataset for augmentation
        transforms: transforms which will be applied to original batch to get both
        left and right output batch.
        transform_left: transform only for left batch
        transform_right: transform only for right batch
        transform_original: transforms which will be applied to save original in batch
        is_target: the flag for selection does dataset return (sample, target) or only sample

    Example:

    .. code-block:: python

        import torchvision
        from torchvision.datasets import CIFAR10

        from catalyst.contrib.data.dataset import SelfSupervisedDatasetWrapper

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                ),
                torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            ]
        )

        cifar_dataset = CIFAR10(root="./data", download=True, transform=None)
        cifar_contrastive = SelfSupervisedDatasetWrapper(cifar_dataset, transforms=transforms)

    .. _`A Simple Framework for Contrastive Learning of Visual Representations`:
        https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        dataset: Dataset,
        transforms: Callable = None,
        transform_left: Callable = None,
        transform_right: Callable = None,
        transform_original: Callable = None,
        is_target: bool = True,
    ) -> None:
        """
        Args:
            dataset: original dataset for augmentation
            transforms: transforms which will be applied to original batch to get both
            left and right output batch.
            transform_left: transform only for left batch
            transform_right: transform only for right batch
            transform_original: transforms which will be applied to save original in batch
            is_target: the flag for selection does dataset return (sample, target) or only sample

        """
        super().__init__()

        if transform_right is not None and transform_left is not None:
            self.transform_right = transform_right
            self.transform_left = transform_left
        elif transforms is not None:
            self.transform_right = transforms
            self.transform_left = transforms
        else:
            raise ValueError(
                "Specify transform_left and transform_right simultaneously or only transforms."
            )
        self.transform_original = transform_original
        self.dataset = dataset
        self.is_target = is_target

    def __len__(self) -> int:
        """Length"""
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """Get item method for dataset

        Args:
            idx: index of the object

        Returns:
            Dict with left agumention (aug1), right agumention (aug2) and target
        """
        if self.is_target:
            sample, target = self.dataset[idx]
        else:
            sample = self.dataset[idx]

        transformed_sample = self.transform_original(sample) if self.transform_original else sample
        aug_1 = self.transform_left(sample)
        aug_2 = self.transform_right(sample)

        if self.is_target:
            return transformed_sample, aug_1, aug_2, target
        return transformed_sample, aug_1, aug_2
