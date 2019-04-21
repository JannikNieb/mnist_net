import numpy as np


class ConvolutionalNet:

    def __init__(self, kernel_size, depth, learning_rate, batch_size):
        # kernels randomized according to normal distribution with mü=0 and sigma = sqrt(number of kernel parameters)
        self.kernel = [np.random.randn(batch_size, kernel_size, kernel_size) * np.sqrt(kernel_size)
                       for x in range(depth)]
        self.kernel_size = kernel_size
        self.depth = depth
        self.layers = []
        self.lRate = learning_rate
        self.pooling_stride = 2
        self.batch_size = batch_size

    def relu(self, input, deriv=False):
        """
        activation function which is used after the convolutional layer. It returns a zero if the value is < 0 and
        the same value if it is > 0
        :param input: Float
        :param deriv: Boolean
        :return: Float
        """
        if not deriv:  # normal ReLU function
            output = max([0, input])
        elif deriv:  # derivative of the ReLU function
            # returns a 1 if the input is greater than zero and a zero otherwise
            output = float(np.greater(input, 0).astype(int))
        return output

    def sig(self, x, deriv=False):
        """
        sigmoid activation function, which is used to normalize values (bring them between 0 and 1) in the
        fully-connected layer
        :param x: Float
        :param deriv: Boolean
        :return: Float
        """
        if deriv == False:  # normal sigmoid function
            return 1 / (1 + np.exp(-x))
        elif deriv:  # derivative of the sigmoid function (for an already squashed value)
            return x * (1 - x)

    def softmax(self, x, deriv=False):
        """
        softmax activation function, which brings all values in the array to a value between 0 and 1. Additionally the
        sum of the array values is 1
        :param x: np.array or list
        :param deriv: Boolean
        :return: list of same size as x
        """
        if not deriv:  # normal softmax function
            return list(np.exp(x) / sum(np.exp(x)))
        elif deriv:  # derivative of the softmax function (for an already squashed value)
            return np.array(x) * (1 - np.array(x))

    def tanh(self, x, deriv=False):
        """
        tanh (hyperbolic tangent) activation function, which brings all values in the array to a value between -1 and 1.
        tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        :param x: np.array or list
        :param deriv: Boolean
        :return: list of same size as x
        """
        if not deriv:  # normal tanh function
            return np.tanh(x)
        if deriv:  # derivative of the tanh function (for an already squashed value)
            return 1 - np.square(x)

    def generate_zero_padding(self, image_count, images):
        """
        generate a padding with the value 0.0 around the outside of the image matrix to keep its size constant after the
        convolutional layer
        :param input_matrix: np.matrix containing 28 * 28 Mnist image
        :return: np. matrix of original image with layer of zeros around the edges
        """
        outputs = []
        padding_size = int((self.kernel_size - 1) / 2)
        image_size = int(np.sqrt(len(images[0])))
        output_matrix = np.zeros((image_size + 2 * padding_size, image_size + 2 * padding_size))
        # create a new matrix of the output size and past the original image in its middle
        for i in range(image_count):
            image = images[i].reshape(image_size, image_size)
            output_matrix[padding_size: -padding_size, padding_size: -padding_size] = image
            outputs.append(output_matrix)
        return outputs

    def conv_layer(self, input_matrix):
        """
        convolutional layer of the network (stride = 1)
        :param input_matrix: list containing {batch_size} np.matrices with zero padding
        :return: {batch_size} lists each containing {depth} np.matrices of size 28 * 28
        """
        input_size = len(input_matrix[0])
        output_size = input_size - self.kernel_size + 1
        output = [[np.zeros((output_size, output_size)) for i in range(self.depth)] for j in range(self.batch_size)]
        for c in range(self.batch_size):
            for x in range(self.depth):
                # k, h for scanning over the input matrix
                for k in range(output_size):
                    for h in range(output_size):
                        # j, i as iterators for Frobenius inner product (here: inner_sum)
                        inner_sum = 0
                        for j in range(self.kernel_size - 1):
                            for i in range(self.kernel_size - 1):
                                    inner_sum += (input_matrix[c][k + j, h + i] * self.kernel[x][c, j, i])
                        output[c][x][k, h] = self.relu(inner_sum, False)
        return output

    def pool(self, inputs):
        """
        max pooling: only filtering out the maximum value of a stride * stride square of the input matrix
        :param inputs: {batch_size} lists each containing {depth} np.matriced of size 28 * 28
        :return: {batch_size} lists each containing {depth} np.matrices of size 14 * 14
        """
        input_size = len(inputs[0][0])
        # testing if the size of the matrix is dividable by 2/ 3 to determine the stride
        if input_size % 2 == 0:
            self.pooling_stride = 2
        elif input_size % 3 == 0:
            self.pooling_stride = 3
        else:
            print("matrix of uneven length: no pooling possible")

        output_size = int(input_size / self.pooling_stride)
        output = [[np.zeros([output_size, output_size]) for i in range(self.depth)] for j in range(self.batch_size)]

        for d in range(self.depth):
            for b in range(self.batch_size):
                for g in range(0, input_size, self.pooling_stride):
                    for h in range(0, input_size, self.pooling_stride):
                        # i, j as iterators for scanning over the pooling field
                        for i in range(self.pooling_stride):
                            for j in range(self.pooling_stride):
                                output[b][d][int(g / 2), int(h / 2)] = max(output[b][d][int(g / 2), int(h / 2)],
                                                                           inputs[b][d][g + i, h + j])
        return output

    def concentration(self, input, backward=False):
        """
        reformation of the list of matrices to a vector before the fully-connected layer/ otherwise in backpropagation
        :param input: list of matrices/ np.array
        :param backward: Boolean
        :return:
        """
        output = []
        if not backward:
            for j in range(self.depth):
                output.extend(input[j].flatten())
            output = np.array(output)
        elif backward:
            output = []
            for i in range(self.batch_size):
                output.append(np.array_split(input[i][-1], self.depth))  # returns list of np.arrays
        return output

    def initialize_fully_connected(self, fc_size, input_image_size, image_count):
        """
        initializing the layers of the fully connected layer. The size of the first fc layer is equal to the size of
        the MNIST image divided by 2 (because of the pooling), the hidden layers all have size 100 * 100 and the output
        layer has size 100 * 10 (for a output of the 10 numbers). The values are normally distributed around 0 with a
        variance of 1 / sqrt(amount of total images)
        :param fc_size: int: Number of hidden layers to be generated
        :param input_image_size: int: size of the image(before pooling)
        :return:
        """
        np.random.seed(0)
        layer_size = self.depth * int(np.sqrt(input_image_size) / 2) ** 2
        # generating a transition layer, a user defined number of hidden layers of size 100 * 100 and an output layer
        self.layers = [np.random.randn(self.batch_size, layer_size, 100) * (1 / np.sqrt(image_count))]
        for j in range(fc_size):
            self.layers.append(np.random.randn(self.batch_size, 100, 100))
        self.layers.append(np.random.randn(self.batch_size, 100, 10))
        return

    def fully_connected(self, input_matrix, concentrate=True):
        """
        multiplies the vector step by step with all the weigth matrices. All the results are stored in the outputs
        vector. The first value has size 588, the following have size 100 and the last value has size 10.
        :param input_matrix: {batch_size} lists each containing {depth} np.matrices of size 14 * 14
        :param concentrate: Boolean
        :return: {batch_size} lists each containing {fc_size} np.arrays
        """
        outputs = [[] for x in range(self.batch_size)]

        for i in range(self.batch_size):
            if concentrate:
                image = np.array(self.concentration(input_matrix[i]))
            else:
                image = input_matrix[i]
            outputs[i].append(image)
            # multiplying the input vector with all the hidden layer matrices
            for j in range(len(self.layers) - 1):
                outputs[i].append(self.tanh(np.dot(np.array(image).T, self.layers[j][i])))
                image = outputs[i][j + 1]
            # softmax activation function for the last layer
            outputs[i].append(self.softmax(np.dot(np.array(image).T, self.layers[j + 1][i])))
        return outputs

    def fc_backprop(self, labels, results):
        """
        adjusts the weights to minimize the error of the hidden layers by using gradient descend
        :param labels: int: correct labels of the training images
        :param results: List of np arrays
        :return:
        """
        hidden_errors = [[] for x in range(self.batch_size)]
        for i in range(self.batch_size):
            target = [0.0] * 10
            target[labels[i]] = 1.0  # list of targets (for all nodes zero, but for the actual label of the image)
            error = target - np.array(results[i][len(self.layers)])  # error of the last layer
            hidden_errors[i].append(error)  # list of all the hidden errors (error of last layer 1st in list!)
            for j in range(len(self.layers)):
                # calculating the hidden errors by multiplying the error of the 1st layer with the hidden layer matrices
                hidden_errors[i].append(np.dot(self.layers[len(self.layers) - (j + 1)][i], error))
                error = hidden_errors[i][j + 1]
                # adjust weights
                if j == 0:
                    # again using the derivative of the softmax function for the results of the last layer
                    hidden_out = self.softmax(results[i][len(self.layers) - j], True)
                else:
                    hidden_out = self.tanh(results[i][len(self.layers) - j], True)
                correction = self.lRate * np.outer((hidden_errors[i][j] * hidden_out),
                                                   results[i][len(self.layers) - (j + 1)]).T
                self.layers[len(self.layers) - (j + 1)][i] += correction
        return hidden_errors

    def pooling_backprop(self, fc_error, results):
        """
        backpropagation of the pooling layer.
        :param fc_error: np.array: error of the fully-connected layer
        :param results: results of the forward pass of the pooling layer and the previous one
        :return: np.array: error after the pooling layer
        """
        # recreating the matrices from the
        # fc_error = np.array_split(fc_error, self.depth)
        input_size = len(results[len(results) - 1][0][0])
        out_size = 2 * input_size
        pooling_error = [[np.zeros((out_size, out_size)) for x in range(self.depth)] for y in range(self.batch_size)]
        for b in range(self.batch_size):
            for h in range(self.depth):
                for i in range(0, out_size, self.pooling_stride):
                    for j in range(0, out_size, self.pooling_stride):
                        for p in range(self.pooling_stride):
                            for q in range(self.pooling_stride):
                                if results[len(results) - 1][b][h][int(i / 2), int(j / 2)] == \
                                        results[len(results) - 2][b][h][i + p, j + q]:
                                    pooling_error[b][h][i + p, j + q] = fc_error[b][h][i + j]
        return pooling_error

    def convolutional_backprop(self, pooling_error, results):
        output_size = self.kernel_size
        convolutional_error = [[np.zeros((output_size, output_size)) for x in range(self.depth)]
                               for y in range(self.batch_size)]
        for b in range(self.batch_size):
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
                                sum += (pooling_error[b][h][i + p, j + q] *
                                        self.relu([results[b][h][i + p, j + q]], True) * self.kernel[h][b][i, j])
                        convolutional_error[b][h][i, j] = sum
                self.kernel[h][b] += self.lRate * convolutional_error[b][h]
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
        for i in range(0, image_count, self.batch_size):
            conv = self.conv_layer(images[i:i + self.batch_size])  # convolutional layer
            # results.append(conv)  # depth lists of size image_size ^ 2
            pooled = self.pool(conv)  # pooling layer
            # results.append(pooled)  # depth lists of size (image_size / 2) ^ 2
            # 1 list of size ((image_size / 2) * depth) ^ 2 with fc_size + 1 inner lists
            results = self.fully_connected(pooled)
            for j in range(self.batch_size):
                target = labels[i + j]
                prediction_list = results[j][-1]  # activations of the output layer
                prediction = prediction_list.index(max(prediction_list))  # node with the highest activation
                # print(target, prediction_list, prediction)
                if target != prediction:
                    error += 1
        accuracy = 100 - (error / image_count * 100)  # calculate part of right guesses (in percent)
        # self.accuracy_list.append(accuracy)
        return error, accuracy
