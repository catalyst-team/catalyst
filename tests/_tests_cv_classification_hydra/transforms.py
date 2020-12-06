from catalyst.contrib.data.cv import Compose, Normalize, ToTensor


def simple_transform():
    """
    Simple Transform

    Returns:
        Compose: transforms

    """
    return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


__all__ = ["simple_transform"]
