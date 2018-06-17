import numpy as np
from mnist_extractor1 import MNISTExtractor


class mnist_net:

    def __init__(self, input_nodes=784, hidden_nodes=100, output_nodes=10):
        self.iNodes = input_nodes
        self.hNodes = hidden_nodes
        self.oNodes = output_nodes
        self.wih = []
        self.who = []

    def sig(self, x, deriv=False):
        """
        sigmoid function, which is used to normalize valules (bring them between 0 and 1)
        :param x: Int/Float
        :param deriv: Boolean
        :return: Float
        """
        if deriv == False:
            return 1 / (1 + np.exp(-x))
        elif deriv:
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def generate_weights(self):
        """
        generates 2 weigth matrices, randomized according to the gaussian normal distribution
        :return:
        """
        self.wih = np.random.normal(0.0, pow(self.iNodes, -0.5), (self.iNodes, self.hNodes))
        self.who = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.oNodes))
        return

    def execute(self, labels, input, image_count):
        for i in range(image_count):
            # calculating the output of the net
            Hresult = self.sig(np.dot(input[i], self.wih))
            Oresult = self.sig(np.dot(Hresult, self.who))

            # calculating the error of this output
            target = [0.0] * 10
            target[labels[i]] = 1.0
            error = 0
            for j in range(10):
                error += np.square(Oresult[j] - target[j])

        return
