import numpy as np


class mnist_net:

    def __init__(self, input_nodes=784, hidden_nodes=100, output_nodes=10, learning_rate=1):
        self.iNodes = input_nodes
        self.hNodes = hidden_nodes
        self.oNodes = output_nodes
        self.lRate = learning_rate
        self.wih = []
        self.who = []

    def sig(self, x, deriv=False):
        """
        sigmoid function, which is used to normalize values (bring them between 0 and 1)
        :param x: Int/Float
        :param deriv: Boolean
        :return: Float
        """
        if deriv == False:
            return 1 / (1 + np.exp(-x))
        elif deriv:
            return x * (1 - x)

    def generate_weights(self):
        """
        generates 2 weight matrices, randomized according to the gaussian normal distribution
        :return:
        """
        np.random.seed(0)
        self.wih = np.random.normal(0.0, pow(self.iNodes, -0.5), (self.iNodes, self.hNodes))
        self.who = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.oNodes))
        return

    def backpropagation(self, labels, input, Oresult, Hresult):
        """

        :param labels:
        :param input:
        :param Oresult:
        :param Hresult:
        :return:
        """
        # calculating the error of the net for output and hidden layer
        target = [0.0] * 10
        target[labels] = 1.0
        error = target - Oresult
        hidden_error = np.dot(self.who, error)

        # adjust weights
        self.who += self.lRate * np.outer((error * self.sig(Oresult, True)), Hresult).T
        self.wih += self.lRate * np.outer((hidden_error * self.sig(Hresult, True)), input).T
        return

    def execute(self, input):
        """
        calculate the outputs of the different layers
        :param input: 784-dimensional np array
        :return: list of 2 np arrays
        """

        outputs = []
        outputs.append(self.sig(np.dot(np.transpose(input), self.wih)))  # Hresult
        outputs.append(self.sig(np.dot(np.transpose(outputs[0]), self.who)))  # Oresult
        return outputs

    def predict(self, labels, images, error):
        target = labels
        prediction_list = list(self.execute(images)[1])  # activations of the output layer
        prediction = prediction_list.index(max(prediction_list))  # node with the highest activation
        # print("target: {}, prediction: {}".format(target, prediction))
        if target != prediction:
            error += 1
        return error
