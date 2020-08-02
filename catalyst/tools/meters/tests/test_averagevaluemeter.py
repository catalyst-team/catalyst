# flake8: noqa
# @TODO: code formatting issue for 20.07 release

import torch

from catalyst.tools import meters


def test_averagevaluemeter():
    """Test for ``catalyst.tools.meters.AverageValueMeter``."""
    meter_instance = meters.AverageValueMeter()

    def batch_generator(length, batch_size=10):
        data = torch.rand(length)
        for i in range(length // batch_size):
            yield data[i * batch_size : (i + 1) * batch_size]
        if length % batch_size:
            yield data[-(length % batch_size) :]

    def test(meter, length, batch_size):
        x2 = torch.zeros(length)
        i = 0
        for batch in batch_generator(length, batch_size):
            bs = batch.shape[0]
            meter.add(batch.mean(), bs)
            x2[i : i + bs] = batch.mean()
            i += bs
        assert torch.allclose(
            torch.tensor((x2.mean(), x2.std())), torch.tensor(meter.value())
        )
        meter.reset()

    confs = ((100, 1), (100, 10), (100, 16), (1024, 53), (10, 16), (100, 100))
    for conf in confs:
        test(meter_instance, *conf)
