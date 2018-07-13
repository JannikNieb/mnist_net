import numpy as np
import json
import os
import struct


class MnistNet:

    def __init__(self, input_nodes=784, hidden_layer_size=[], output_nodes=10, learning_rate=1, accuracy=0):
        self.iNodes = input_nodes
        self.hNodes = hidden_layer_size
        self.oNodes = output_nodes
        self.lRate = learning_rate
        self.accuracy = accuracy
        self.layers = []

    def generate_weights(self):
        """
        generating matrices for every layer, randomized according to the gaussian normal distribution
        :return: list of np matrices
        """
        np.random.seed(0)
        for i in range(len(self.hNodes) - 1):
            self.layers.append(np.random.normal(0.0, pow(self.hNodes[i], -0.5), (self.hNodes[i], self.hNodes[i + 1])))
            # self.layers.append(np.random.rand(self.hNodes[i], self.hNodes[i + 1]))

    def sig(self, x, deriv=False):
        """
        sigmoid activation function, which is used to normalize values (bring them between 0 and 1)
        :param x: Float
        :param deriv: Boolean
        :return: Float
        """
        if deriv == False:  # normal sigmoid function
            return 1 / (1 + np.exp(-x))
        elif deriv:  # derivative of the sigmoid function (for an already squashed value)
            return x * (1 - x)

    def backpropagation(self, labels, input, results):
        """
        adjusts the weights to minimize the error of the output- and hidden layer
        :param labels: Integer
        :param input: np-array (784 dimensions)
        :param Oresult: np-arrray (10 dimensions)
        :param Hresult: np-array (hNodes dimensions)
        :return:
        """
        # calculating the error of the net for output and hidden layer
        target = [0.0] * 10
        target[labels] = 1.0  # list of targets (for all nodes zero, but for the actual label of the image)
        error = target - results[len(self.layers) - 1]  # error of the last layer
        hidden_errors = [error]  # list of all the hidden errors (error of last layer 1st in list!)
        for i in range(len(self.layers) - 1):
            hidden_errors.append(np.dot(self.layers[len(self.layers) - (i + 1)], error))
            error = hidden_errors[i + 1]
            # adjust weights
            self.layers[len(self.layers) - (i + 1)] += self.lRate * np.outer((hidden_errors[i] *
                            self.sig(results[len(self.layers) - (i + 1)], True)), results[len(self.layers) - (i + 2)]).T
        return

    def execute(self, input):
        """
        calculate the outputs of the different layers
        :param input: 784-dimensional np array
        :return: list of np arrays
        """
        outputs = []
        for i in range(len(self.layers)):
            outputs.append(self.sig(np.dot(input.T, self.layers[i])))  # Hresult
            input = outputs[i]
        return outputs

    def calculate_accuracy(self, labels, images, image_count):
        """
        Calculates the number of wrong predictions of the net and it´s accuracy
        :param labels: Integer
        :param images: np-array (784-dimensional)
        :param image_count: Integer
        :return:
        """
        error = 0
        for i in range(image_count):
            target = labels[i]
            prediction_list = list(self.execute(images[i])[len(self.layers) - 1])  # activations of the output layer
            prediction = prediction_list.index(max(prediction_list))  # node with the highest activation
            if target != prediction:
                error += 1
            self.accuracy = 100 - (error / image_count * 100)  # calculate part of right guesses (in percent)
        return error

    def enter_image_count(self, max):
        """
        Shows the user how many images are in the data set and asks how many should be loaded
        :param max: Integer
        :return: Integer
        """
        print("There are {} images available.".format(max))
        while True:
            user_in = input("How many images would you like to load? (Enter '*' to load all) > ")
            if user_in == "*":
                custom_image_count = max
                break
            elif int(user_in) <= max:  # only continues if there are as many images in the set as the user wants to load
                custom_image_count = int(round(float(user_in)))
                break
            else:
                print("\033[2;31;40m" + "Error: Not that many training images available!" + "\033[0m")
        return custom_image_count

    def save_json_results(self, foldername, custom_image_count, epochs, layer_size):
        """
        saves the score and important parameters of the net in a json-file so they can be displayed with a seperate
        web app (table.py)
        :param foldername: String
        :param custom_image_count: Integer
        :param epochs: Integer
        :param layer_size: List of Integers
        :return:
        """
        # returns number of all previously existing json-files in the folder + 1
        j = len([x for x in os.scandir("net_data") if x.is_file() and x.name.endswith(".json")]) + 1

        # dict with all the data to be saved (score + important parameters)
        data = {"id": j, "accuracy": self.accuracy, "learning_rate": self.lRate, "hidden_layers": len(layer_size) - 2,
                "trained_images:": custom_image_count, "epochs": epochs}
        for i in range(len(layer_size) - 2):
            data[f"nodes in layer {i + 1}"] = layer_size[i + 1]

        # writing the data dict into a json-file
        filename = foldername + f"/score{j}.json"
        print(f"Writing data to file: {filename}")
        with open(filename, 'w') as file:
            json.dump(data, file)
        return

    def save(self, foldername):
        """
        saves the matrices of the neural net in a seperate file
        :param foldername: String
        :return:
        """
        # returns number of all previously existing json-files in the folder + 1
        j = len([x for x in os.scandir("net_data") if x.is_file() and not x.name.endswith(".json")]) + 1
        filename = foldername + f"/net{j}"
        with open(filename, 'wb') as file:
            file.write(struct.pack('<I', len(self.hNodes)))
            for i in self.hNodes:
                file.write(struct.pack('<I', i))

            # saving all the (flattened) hidden matrices and a hex-id (according to their position in the net) in a file
            for i in range(len(self.hNodes) - 1):
                array = self.layers[i].flatten()
                file.write(struct.pack('f' * len(array), *array))
            file.close()
            return

    def load(self, filename):
        """
        Loading a previously saved net from the file
        :param filename: String
        :return:
        """
        with open(filename, 'rb') as file:
            hidden_layers = np.fromfile(file, dtype='<u4', count=1)[0]
            for i in range(hidden_layers):
                self.hNodes.append(np.fromfile(file, dtype='<u4', count=1)[0])

            # reconstructing the matrices
            for i in range(len(self.hNodes) - 1):
                matrix_size = self.hNodes[i] * self.hNodes[i + 1]
                linear_list = np.fromfile(file, dtype='f', count=matrix_size)
                self.layers.append(np.reshape(linear_list, (self.hNodes[i], self.hNodes[i + 1])))

            file.close()
            return
