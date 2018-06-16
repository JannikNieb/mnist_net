import numpy as np

class mnist_net:

    def __init__(self, input_notes=784, hidden_notes=100, output_notes=10):
        self.input_notes = input_notes
        self.hidden_notes = hidden_notes
        self.output_notes = output_notes
        self.wih = []
        self.who = []

    # sigmoid function, which is used to normalized valules
    def sig(self, x, deriv=False):
        if deriv == False:
            return 1 / (1 + np.exp(-x))
        elif deriv == True:
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


