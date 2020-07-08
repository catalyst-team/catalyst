import torch

from catalyst.contrib.dl.callbacks.cmc_callback import CMCScoreCallback
from catalyst.dl import SupervisedRunner
from catalyst.utils import set_global_seed


class QueryGalleryDataset(torch.utils.data.Dataset):
    """Dummy dataset"""

    def __init__(self):
        """Init random dataset"""
        set_global_seed(42)
        self.embeddings = torch.randn((10, 10))
        dist = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        self.labels = dist.sample((10, 1))

    def __getitem__(self, item):
        """Get key-value object"""
        output_dict = {}
        output_dict["features"] = self.embeddings[item]
        output_dict["labels"] = self.labels[item]
        output_dict["query"] = item > 4
        return output_dict

    def __len__(self):
        """Length"""
        return self.embeddings.shape[0]


def test_pipeline():
    """Test if simple pipeline works"""
    net = torch.nn.Linear(10, 4)
    dataset = QueryGalleryDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    runner = SupervisedRunner(output_key="embeddings")
    runner.train(
        model=net,
        loaders={"valid": dataloader},
        main_metric="cmc_1",
        callbacks={"cmc": CMCScoreCallback(topk_args=(1, 2))},
        check=True,
    )
    assert True
