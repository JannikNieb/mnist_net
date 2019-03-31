from cnn_scratch import ConvolutionalNet
from mnist_extractor1 import MNISTExtractor

# defining the parameters of the convolutional net
kernel_size = 3  # needs to be uneven!!!
depth = 3  # int(input("How many kernels should be used? > "))
fc_size = 2  # int(input("How many fully connected layers should be used? > "))
learning_rate = 0.3  # input("Enter the learning rate! > ")
epochs = 5


# initializing the network and the mnist image extractor
mne = MNISTExtractor()
cnn = ConvolutionalNet(kernel_size, depth, learning_rate)

train_labels = mne.extractLabels("mnist_data/train_labels.bin")
custom_image_count = cnn.enter_image_count(len(train_labels))
train_images = mne.extractImages("mnist_data/train_images.bin", custom_image_count)

# size of the first fc layer is equal to the size of the MNIST image divided by 2 (because of the pooling)
image_size = len(train_images[0])
cnn.initialitze_fully_conected(fc_size, image_size)

for j in range(epochs):
    for i in range(custom_image_count):
        results = []
        """train_images[i] = train_images[i].reshape(28, 28)  # reshaping the image into a 28 * 28 matrix
        padded_image = cnn.generate_zero_padding(train_images[i])  # adding a zero padding around it
        conv = [cnn.conv_layer(padded_image, x) for x in range(cnn.depth)]
        results.append(conv)  # depth lists of image_size ^ 2
        pooled = [cnn.pool(conv[x]) for x in range(cnn.depth)]  # pooling layer
        results.append(pooled)  # depth lists of (image_size / 2) ^ 2
        # 1 list of size ((image_size / 2) * depth) ^ 2 with fc_size + 1 inner lists
        results.append(cnn.fully_connected(pooled, fc_size))"""
        # results.append(cnn.fully_connected(train_images[i]))
        cnn.fc_backprop(train_labels[i], cnn.fully_connected(train_images[i]))
        # pool_error = cnn.pooling_backprop(fc_error, results[len(results) - 3:len(results) - 1:])
        # cnn.convolutional_backprop(pool_error, results[0])
        # print(cnn.convolutional_backprop(pool_error, results[0]))
        # results = list(results[-1][-1])
        # print(results, train_labels[i])
        # print(results.index(max(results)), train_labels[i])
    print(cnn.calculate_accuracy(train_labels, train_images, custom_image_count))
    print("---")
