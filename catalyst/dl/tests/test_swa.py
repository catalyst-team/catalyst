import os
from pathlib import Path
import shutil
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from catalyst.dl.utils.swa import generate_averaged_weights

sys.path.append(".")


class Net(nn.Module):
    def __init__(self, init_weight=4):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.fc.weight.data.fill_(init_weight)
        self.fc.bias.data.fill_(init_weight)

    def forward(self, x):
        x = self.fc(x)
        return x


class TestSwa(unittest.TestCase):
    def setUp(self):
        net1 = Net(init_weight=2)
        net2 = Net(init_weight=4)
        os.mkdir("./checkpoints")
        torch.save(net1.state_dict(), "./checkpoints/net1.pth")
        torch.save(net2.state_dict(), "./checkpoints/net2.pth")

    def tearDown(self):
        shutil.rmtree("./checkpoints")

    def test_averaging(self):
        generate_averaged_weights(
            logdir=Path("./"),
            models_mask="net*",
            save_path=Path("./checkpoints"),
        )
        model = Net()
        model.load_state_dict(torch.load("./checkpoints/swa_weights.pth"))

        self.assertEqual(int(model.fc.weight.data[0][0]), 3)
        self.assertEqual(int(model.fc.weight.data[0][1]), 3)
        self.assertEqual(int(model.fc.bias.data[0]), 3)


if __name__ == "__main__":
    unittest.main()
