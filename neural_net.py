import numpy as np
import json
import os
import struct


class MnistNet:

    def __init__(self, input_size=[28, 28, 3], hidden_layer_size=[], output_nodes=10, learning_rate=1):
        self.input_size = input_size
        self.hNodes = hidden_layer_size
        self.oNodes = output_nodes
        self.lRate = learning_rate
        self.accuracy_list = []
        self.layers = []

    def generate_weights(self):
        """
        generating matrices for every layer, randomized according to the gaussian normal distribution
        :return: List of np matrices
        """
        np.random.seed(0)
        for i in range(len(self.hNodes) - 1):
            self.layers.append(np.random.normal(0.0, pow(self.hNodes[i], -0.5), (self.hNodes[i], self.hNodes[i + 1])))

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

    def backpropagation(self, labels, results):
        """
        adjusts the weights to minimize the error of the hidden layers
        :param labels: Integer
        :param results: List of np arrays
        :return:
        """
        # calculating the error of the net for output and hidden layer
        target = [0.0] * 10
        target[labels] = 1.0  # list of targets (for all nodes zero, but for the actual label of the image)
        error = target - results[len(self.layers) - 1]  # error of the last layer
        hidden_errors = [error]  # list of all the hidden errors (error of last layer 1st in list!)
        for i in range(len(self.layers) - 1):
            # calculating the hidden errors by multiplying the error of the 1st layer with the hidden layer matrices
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
        :return: List of np arrays
        """
        outputs = []
        # multiplying the input vector with all the hidden layer matrices
        for i in range(len(self.layers)):
            outputs.append(self.sig(np.dot(input.T, self.layers[i])))
            input = outputs[i]
        return outputs

    def cnn_generate_kernel(self, kernel_size):
        """
        generate randomized kernel
        :param kernel_size: Integer
        :return: np.matrix of size (kernel_size ^ 2)
        """
        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i][j] = np.random.rand(1)
        return kernel

    def cnn_conv_layer(self, kernel, input_matrix):
        """
        convolutional layer of the network (padding = 0, stride = 1)
        :param kernel: np.matrix
        :param input_matrix: np.matrix
        :return: np.matrix of size ((input_size - kernel_size + 1) ^ 2)
        """
        kernel_size = len(kernel)
        input_size = len(input_matrix)
        output_size = input_size - kernel_size + 1
        output = np.zeros((output_size, output_size))
        # k, h for scanning over the input matrix
        for k in range(output_size):
            for h in range(output_size):
                # j, i for Frobenius inner product (here: sum)
                temp = 0
                for j in range(kernel_size):
                    for i in range(kernel_size):
                            temp += (kernel[j, i] * input_matrix[k + j, h + i])
                output[k, h] = temp
        return output

    def cnn_relu(self, input):
        """
        activition function which returns a zero if the value is < 0 and the same value if it is > 0
        :param input: np.matrix
        :return: np.matrix of same size
        """
        output = input
        for i in range(len(input)):
            for j in range(len(input)):
                output[i, j] = max([0, input[i, j]])
        return output

    def cnn_pool(self, input_matrix):
        """
        reducing the sample size by only further using the highest value in a 2 * 2 square
        (3 * 3 if otherwise not possible)
        :param input_matrix: np.matrix
        :return: output: np.matrix of size ((input_matrix / 2) ^ 2)
        """
        # trying a padding of 2, if not possible using one of 3, if this isn´t possible either exiting the program
        if len(input_matrix) % 2 == 0:
            stride = 2
            output_size = int(len(input_matrix) / stride)
            print("pooling out size:", output_size)
        elif len(input_matrix) % 3 == 0:
            stride = 3
            output_size = int(len(input_matrix) / stride)
        else:
            print("matrix of uneven lenght: no pooling possible")
            exit()

        print("using a padding of", stride)

        output = np.zeros((output_size, output_size))
        for i in [x for x in range(len(output))[::2]]:
            for j in [x for x in range(len(output))[::2]]:
                output[i, j] = max(input_matrix[i, j].flatten())
        return output

    def calculate_accuracy(self, labels, images, image_count):
        """
        Calculates the number of wrong predictions of the net and its accuracy
        :param labels: Integer
        :param images: np-array (784-dimensional)
        :param image_count: Integer
        :return error: Integer
        :return accuracy: Float
        """
        error = 0
        for i in range(image_count):
            target = labels[i]
            prediction_list = list(self.execute(images[i])[len(self.layers) - 1])  # activations of the output layer
            prediction = prediction_list.index(max(prediction_list))  # node with the highest activation
            if target != prediction:
                error += 1
        accuracy = 100 - (error / image_count * 100)  # calculate part of right guesses (in percent)
        self.accuracy_list.append(accuracy)
        return error, accuracy

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
            else:
                try:  # return error when something other than a number is entered
                    if int(round(float(user_in))) <= max:  # only continues if there are as many images in the set as the user wants to load
                        custom_image_count = int(round(float(user_in)))
                        break
                    else:
                        print("\033[2;31;40m" + "Error: Not that many training images available!" + "\033[0m")

                except ValueError:
                    print("\033[2;31;40m" + "Error: Please enter a integer!" + "\033[om")
        return custom_image_count

    def save_json_results(self, foldername, custom_image_count, epochs, layer_size, accuracy):
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
        data = {"id": j, "accuracy": accuracy, "learning_rate": self.lRate, "hidden_layers": len(layer_size) - 2,
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
        j = len([x for x in os.scandir("net_data") if x.is_file() and x.name.endswith(".json")]) + 1
        filename = foldername + f"/net{j}"
        # writing amount of hidden and number of nodes in each hidden layer to file to allow easy reconstruction
        with open(filename, 'wb') as file:
            file.write(struct.pack('<I', len(self.hNodes)))
            for i in self.hNodes:
                file.write(struct.pack('<I', i))

            # saving all the (flattened) hidden matrices in a file
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

    def cancel_training(self):
        """
        Returns False (und thus cancels the training if no (or negative) improvement occurred over the last 7 epochs
        :return: Boolean
        """
        delta_accuracies = 0
        if len(self.accuracy_list) >= 10:  # starts checking after 10 epochs
            # adding the change in the accuracy over the last 7 epochs and returning Fasle if it is less then 0.3
            for i in range(7):
                delta_accuracies += self.accuracy_list[len(self.accuracy_list) - (i + 1)] - \
                                    self.accuracy_list[len(self.accuracy_list) - (i + 2)]
            if delta_accuracies <= 0.3:
                print("Training canceled because of no (or negative) improvement in last 7 epochs")
                return False
            else:
                return True
        else:
            return True
