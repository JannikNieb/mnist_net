import numpy as np


class ConvolutionalNet:

    def __init__(self, kernel_size, depth, learning_rate):
        # kernels randomized according to normal distribution with mü=0 and sigma=sqrt(number of kernel parameters)
        self.kernel = [np.random.normal(0.0, kernel_size, (kernel_size, kernel_size)) for x in range(depth)]
        self.kernel_size = kernel_size
        self.depth = depth
        self.layers = []
        self.lRate = learning_rate
        self.pooling_stride = 2

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

    def generate_zero_padding(self, input_matrix):
        """
        generate a padding with the value 0.0 around the outside of the image matrix to keep its size after the
        convolutional layer constant
        :param input_matrix: np.matrix containing 28 * 28 Mnist image
        :return: np. matrix of original image with layer of zeros around the edges
        """
        padding_size = int((self.kernel_size - 1) / 2)
        output_matrix = np.zeros((len(input_matrix) + 2 * padding_size, len(input_matrix) + 2 * padding_size))
        output_matrix[padding_size: -padding_size, padding_size: -padding_size] = input_matrix
        return output_matrix

    def relu(self, input, deriv):
        """
        activition function which returns a zero if the value is < 0 and the same value if it is > 0
        :param input: np.matrix
        :return: np.matrix of same size
        """
        """output = input
        for i in range(len(input)):
            for j in range(len(input)):
                if not deriv:
                    output[i, j] = max([0, input[i, j]])
                else:
                    output[i, j] = np.greater(input[i, j], 0).astype(int)"""
        if not deriv:
            output = max([0, input])
        else:
            output = np.greater(input, 0).astype(int)
        return output

    def conv_layer(self, input_matrix, iteration):
        """
        convolutional layer of the network (padding = 0, stride = 1)
        :param input_matrix: np.matrix with zero padding
        :param iteration: integer: number of the kernel
        :return: np.matrix of size 28 * 28
        """
        input_size = len(input_matrix)
        # convolutional layer
        output_size = input_size - self.kernel_size + 1
        output = np.zeros((output_size, output_size))
        # k, h for scanning over the input matrix
        for k in range(output_size):
            for h in range(output_size):
                # j, i for Frobenius inner product (here: sum)
                sum = 0
                for j in range(self.kernel_size - 1):
                    for i in range(self.kernel_size - 1):
                            sum += (input_matrix[k + j, h + i] * self.kernel[iteration][j, i])
                output[k, h] = self.relu(sum, False)
        return output

    def pool(self, input_matrix):
        """
        max pooling: only filtering out the maximum value of a stride * stride square of the input matrix
        :param input_matrix: np.matrix of size 28 * 28
        :return: np.matrix of size 14 * 14
        """
        input_size = len(input_matrix[0])
        # testing if the size of the matrix is dividable by 2/ 3 to determine the stride
        if input_size % 2 == 0:
            self.pooling_stride = 2
        elif input_size % 3 == 0:
            self.pooling_stride = 3
        else:
            print("matrix of uneven length: no pooling possible")

        output_size = int(input_size / self.pooling_stride)
        output = np.zeros([output_size, output_size])
        for i in range(0, input_size, self.pooling_stride):
            for j in range(0, input_size, self.pooling_stride):
                # scaling the original matrix to squares of size stride^2 and adding their max value to an output matrix
                pooling_field = input_matrix[i:(i + self.pooling_stride), j:(j + self.pooling_stride)]
                output[int(i / self.pooling_stride), int(j / self.pooling_stride)] = pooling_field.max()
                # output[x, y] = np.max(output[x, y], input_matrix[x + i, y + j])
        return output

    def initialitze_fully_conected(self, fc_size, input_image_size):
        np.random.seed(0)
        layer_size = self.depth * int(np.sqrt(input_image_size) / 2) ** 2
        # generating a transition layer, a user defined number of hidden layers of size 100 * 100 and an output layer
        self.layers = [np.random.randn(784, 100)]  # layer_size
        for i in range(fc_size):
            self.layers.append(np.random.randn(100, 100))
        self.layers.append(np.random.randn(100, 10))

    def fully_connected(self, input_matrix):
        """image = []
        for i in range(self.depth):
            image.extend(input_matrix[i].flatten())
        image = np.array(image)"""
        image = np.array(input_matrix)
        outputs = [image]
        # multiplying the input vector with all the hidden layer matrices
        for i in range(len(self.layers)):
            outputs.append(self.sig(np.dot(image.T, self.layers[i])))
            image = outputs[i + 1]
        return outputs

    def fc_backprop(self, labels, results):
        """
        adjusts the weights to minimize the error of the hidden layers
        :param labels: Integer
        :param results: List of np arrays
        :return:
        """
        target = [0.0] * 10
        target[labels] = 1.0  # list of targets (for all nodes zero, but for the actual label of the image)
        error = target - results[len(self.layers)]  # error of the last layer
        hidden_errors = [error]  # list of all the hidden errors (error of last layer 1st in list!)
        for i in range(len(self.layers)):
            # calculating the hidden errors by multiplying the error of the 1st layer with the hidden layer matrices
            hidden_errors.append(np.dot(self.layers[len(self.layers) - (i + 1)], error))
            error = hidden_errors[i + 1]
            # adjust weights
            correction = self.lRate * np.outer((hidden_errors[i] * self.sig(results[len(self.layers) - (i)], True)),
                                               results[len(self.layers) - (i + 1)]).T
            self.layers[len(self.layers) - (i + 1)] -= correction
        return

        # calculating the error of the net for output and hidden layer
        """target = [0.0] * 10
        target[labels] = 1.0  # list of targets (for all nodes zero, but for the actual label of the image)
        error = target - results[-1]  # error of the last layer
        hidden_errors = [error]  # list of all the hidden errors (error of last layer 1st in list!)
        for i in range(len(self.layers) - 1):
            # calculating the hidden errors by multiplying the error of the 1st layer with the hidden layer matrices
            hidden_errors.append(np.dot(self.layers[len(self.layers) - (i + 1)], error))
            error = hidden_errors[i + 1]
            # adjust weights
            correction = self.lRate * np.outer((hidden_errors[i] *
                            self.sig(results[len(self.layers) - (i + 1)], True)), results[len(self.layers) - (i + 2)]).T
            self.layers[len(self.layers) - (i + 1)] += correction
        return error"""

    def pooling_backprop(self, fc_error, results):
        fc_error = np.array_split(fc_error, self.depth)
        input_size = len(results[len(results) - 1][0])
        output_size = 2 * input_size
        pooling_error = [np.zeros((output_size, output_size)) for x in range(self.depth)]
        for h in range(self.depth):
            for i in range(0, output_size, self.pooling_stride):
                for j in range(0, output_size, self.pooling_stride):
                    for p in range(self.pooling_stride):
                        for q in range(self.pooling_stride):
                            if results[len(results) - 1][h][int(i / 2), int(j / 2)] == \
                                    results[len(results) - 2][h][i + p, j + q]:
                                pooling_error[h][i + p, j + q] = fc_error[h][i + j]
        return pooling_error

    def convolutional_backprop(self, pooling_error, results):
        output_size = self.kernel_size
        convolutional_error = [np.zeros((output_size, output_size)) for x in range(self.depth)]
        for h in range(self.depth):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    """for p in range(int(-1 * (self.kernel_size - 1) / 2), int((self.kernel_size - 1) / 2)):
                        for q in range(int(-1 * (self.kernel_size - 1) / 2), int((self.kernel_size - 1) / 2)):
                            convolutional_error[i, j] = np.rot90(self.kernel[h][i - p, j - q], 2) * pooling_error[h][i, j] * \
                                                       self.relu(results[0][h], True)"""
                    sum = 0
                    for p in range(self.kernel_size - 1):
                        for q in range(self.kernel_size - 1):
                            sum += (pooling_error[h][i + p, j + q] * self.relu([results[h][i + p, j + q]], True) * self.kernel[h][i, j])
                    convolutional_error[h][i, j] = sum
            self.kernel[h] += convolutional_error[h]
                            # convolutional_error = pooling_error[h] * np.rot90(self.kernel[h], 2) * self.relu(results[0][h], True)

        return convolutional_error

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
                    # only continues if there are as many images in the set as the user wants to load
                    if int(round(float(user_in))) <= max:
                        custom_image_count = int(round(float(user_in)))
                        break
                    else:
                        print("\033[2;31;40m" + "Error: Not that many training images available!" + "\033[0m")

                except ValueError:
                    print("\033[2;31;40m" + "Error: Please enter a integer!" + "\033[om")
        return custom_image_count

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
            prediction_list = list(self.fully_connected(images[i])[-1])  # activations of the output layer
            prediction = prediction_list.index(max(prediction_list))  # node with the highest activation
            # print(target, prediction_list, prediction)
            if target != prediction:
                error += 1
        accuracy = 100 - (error / image_count * 100)  # calculate part of right guesses (in percent)
        # self.accuracy_list.append(accuracy)
        return error, accuracy
