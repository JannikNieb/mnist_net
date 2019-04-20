from cnn_scratch import ConvolutionalNet
from mnist_extractor1 import MNISTExtractor

# defining the parameters of the convolutional net
kernel_size = 3  # needs to be uneven!!!
depth = int(input("How many kernels should be used? > "))
fc_size = 1  # int(input("How many fully connected layers should be used? > "))
learning_rate = float(input("Enter the learning rate! > "))
epochs = int(round(float(input("How many epochs should be executed? > "))))
batch_size = 5


# initializing the network and the mnist image extractor
mne = MNISTExtractor()
cnn = ConvolutionalNet(kernel_size, depth, learning_rate, batch_size)

# extracting the training labels and images
train_labels = mne.extractLabels("mnist_data/train_labels.bin")
custom_image_count = int(round(cnn.enter_image_count(len(train_labels)) / batch_size) * batch_size)
train_images = mne.extractImages("mnist_data/train_images.bin", custom_image_count)

# initializing the fully connected layer of the net
image_size = len(train_images[0])
cnn.initialize_fully_connected(fc_size, image_size, custom_image_count)

padded_image = cnn.generate_zero_padding(custom_image_count, train_images)  # adding a zero padding around the images

for j in range(epochs):
    for i in range(0, custom_image_count, cnn.batch_size):
        # forward pass
        results = []
        conv = cnn.conv_layer(padded_image[i:i + cnn.batch_size])  # convolutional layer
        results.append(conv)  # depth lists of size image_size ^ 2
        pooled = cnn.pool(conv)  # pooling layer
        results.append(pooled)  # depth lists of size (image_size / 2) ^ 2
        # 1 list of size ((image_size / 2) * depth) ^ 2 with fc_size + 1 inner lists
        results.append(cnn.fully_connected(pooled))

        # backpropagation
        fc_error = cnn.fc_backprop(train_labels[i:i + cnn.batch_size], results[-1])
        # pool_error = cnn.pooling_backprop(cnn.concentration(fc_error), results[len(results) - 3:len(results) - 1:])
        # cnn.convolutional_backprop(pool_error, results[0])
    train_error, accuracy = cnn.calculate_accuracy(train_labels, padded_image, custom_image_count)
    print("Epoch {}: {} error: {} {} accuracy: {}%".format(j + 1, " " * abs(7 - len(str(j + 1))), train_error,
                                                           " " * abs((7 - len(str(train_error)))), accuracy))
