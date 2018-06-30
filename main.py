from neural_net import MnistNet
from mnist_extractor1 import MNISTExtractor

mne = MNISTExtractor()

# print colored headline
print("\n" + """          ---Welcome to---
A NEURAL NET RECOGNISING HANDWRITTEN DIGITS
    ---based on the MNIST dataset--- 
""")

# define base parameters of the net
print("\n" + "Your net concists of 784 input nodes (28 * 28 pixel image) and 10 output nodes (10 numbers).")
hidden_nodes = int(round(float(input("Enter the number of hidden nodes! > "))))
learning_rate = float(input("Enter the learning rate! > "))

nn = MnistNet(hidden_nodes=hidden_nodes, learning_rate=learning_rate)

# Loading the MNIST training data set
print("\n" + "Loading MNIST training data labels...")
train_labels = mne.extractLabels("data/train_labels.bin")
print("Your training set contains", len(train_labels), "images")
# enter number of training images and only continue if this many training images are actually in the set
while True:
    custom_image_count = int(round(float(input("How many images would you like to load? > "))))
    if custom_image_count <= len(train_labels):
        break
    else:
        print("\033[2;31;40m" + "Error: Not that many training images available!" + "\033[0m")

print("Loading {} MNIST training images...".format(custom_image_count))
train_images = mne.extractImages("data/train_images.bin", custom_image_count + 1)

# Executing the net
epochs = int(round(float(input("How many epochs should be executed? > "))))

print("\n" + "Executing neural_net")
print("This could take some time...")

for epoch in range(epochs):
    train_error = 0
    for i in range(custom_image_count - 1):
        Hresult = nn.execute(train_images[i])[0]
        Oresult = nn.execute(train_images[i])[1]
        nn.backpropagation(train_labels[i], train_images[i], Oresult, Hresult)
    train_error, train_accuracy = nn.calculate_accuracy(train_labels, train_images, custom_image_count)
    print("Epoch {}:     error: {} {} accuracy: {}%".format(epoch + 1, train_error, " " * abs((7 - len(str(train_error)))), train_accuracy))

# Testing the net on the MNIST test data sets
while True:
    user_in = input("\n" + "Would you like to test your neural net on the MNIST test dataset? (y/n) > ")
    if user_in == "y":
        test_error = 0
        test_labels = mne.extractLabels("data/test_labels.bin")
        print("There are {} test images available.".format(len(test_labels)))
        custom_test_image_count = int(round(float(input("Enter the number of images you would like to load? > "))))
        test_images = mne.extractImages("data/test_images.bin", custom_test_image_count)
        test_error, test_accuracy = nn.calculate_accuracy(test_labels, test_images, custom_test_image_count)
        print("error: {}    accuracy: {}".format(test_error, test_accuracy))
        break
    elif user_in == "n":
        break
    else:
        print("\033[2;31;40m" + "Error: Invalid input! Enter either 'y' to continue or 'n' to quit" + "\033[0m")
