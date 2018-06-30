import numpy as np
from neural_net import mnist_net

nn = mnist_net()


def test_sigmoid():
    assert nn.sig(0) == 0.5
    assert nn.sig(5) == 0.9933071490757153
    assert nn.sig(0.5, True) == 0.25
