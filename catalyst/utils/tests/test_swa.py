# flake8: noqa
import os
from pathlib import Path
import shutil
import unittest

import torch
import torch.nn as nn

from catalyst.utils.swa import get_averaged_weights_by_path_mask


class Net(nn.Module):
    """Dummy network class."""

    def __init__(self, init_weight=4):
        """Initialization of network and filling it with given numbers."""
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.fc.weight.data.fill_(init_weight)
        self.fc.bias.data.fill_(init_weight)


class TestSwa(unittest.TestCase):
    """Test SWA class."""

    def setUp(self):
        """Test set up."""
        net1 = Net(init_weight=2.0)
        net2 = Net(init_weight=5.0)
        os.mkdir("./checkpoints")
        torch.save(net1.state_dict(), "./checkpoints/net1.pth")
        torch.save(net2.state_dict(), "./checkpoints/net2.pth")

    def tearDown(self):
        """Test tear down."""
        shutil.rmtree("./checkpoints")

    def test_averaging(self):
        """Test SWA method."""
        weights = get_averaged_weights_by_path_mask(logdir=Path("./"), path_mask="net*")
        torch.save(weights, str("./checkpoints/swa_weights.pth"))
        model = Net()
        model.load_state_dict(
            torch.load("./checkpoints/swa_weights.pth", map_location=lambda storage, loc: storage)
        )
        self.assertEqual(float(model.fc.weight.data[0][0]), 3.5)
        self.assertEqual(float(model.fc.weight.data[0][1]), 3.5)
        self.assertEqual(float(model.fc.bias.data[0]), 3.5)


if __name__ == "__main__":
    unittest.main()
