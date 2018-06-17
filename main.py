from neural_net import mnist_net
from mnist_extractor1 import MNISTExtractor

mne = MNISTExtractor()
nn = mnist_net()

# print colored headline
print("\033[2;31;40m" + """          ---Welcome to---
A NEURAL NET RECOGNISING HANDWRITTEN DIGITS
    ---based on the MNIST dataset--- 
""" + "\033[0m")

nn.generate_weights()

print("Loading MNIST training data labels...")
train_labels = mne.extractLabels("data/train_labels.bin")

print("Your trainig set contains", len(train_labels), "images")
print("How many would you like to load?")

# enter number of training images and only continue if this many traing images are actually in the set
while True:
    custom_image_count = int(input("Enter a number: "))
    if custom_image_count <= len(train_labels):
        break
    else:
        print("Not that many training images available!")

print("Loading {} MNIST training images".format(custom_image_count))
train_images = mne.extractImages("data/train_images.bin", custom_image_count)

print("Executing neural_net")
nn.execute(train_labels, train_images, custom_image_count)
