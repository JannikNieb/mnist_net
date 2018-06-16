from neural_net import mnist_net

nn = mnist_net()

def test_sigmoid():
    assert nn.sig(0) == 0.5
    assert nn.sig(0, True) == 0.25