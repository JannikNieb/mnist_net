from neural_net import mnist_net
from mnist_extractor1 import MNISTExtractor

mne = MNISTExtractor()
nn = mnist_net()

# print colored headline
print("""          ---Welcome to---
A NEURAL NET RECOGNISING HANDWRITTEN DIGITS
    ---based on the MNIST dataset--- 
""" )

nn.generate_weights()

print("Loading MNIST training data labels...")
train_labels = mne.extractLabels("data/train_labels.bin")

print("Your training set contains", len(train_labels), "images")

# enter number of training images and only continue if this many training images are actually in the set
while True:
    custom_image_count = int(round(float(input("How many images would you like to load? > "))))
    if custom_image_count <= len(train_labels):
        break
    else:
        print("\033[2;31;40m" + "Error: Not that many training images available!" + "\033[0m")

print("Loading {} MNIST training images".format(custom_image_count))
train_images = mne.extractImages("data/train_images.bin", custom_image_count + 1)

epochs = int(round(float(input("How many epochs should be executed? > "))))

print("\n" + "Executing neural_net")
print("This could take some time...")

for epoch in range(epochs):
    train_error = 0
    for i in range(custom_image_count - 1):
        Hresult = nn.execute(train_images[i])[0]
        Oresult = nn.execute(train_images[i])[1]
        nn.backpropagation(train_labels[i], train_images[i], Oresult, Hresult)
        train_error = nn.predict(train_labels[i], train_images[i], train_error)
    print("Epoch {}:    error:{}".format(epoch + 1, train_error))

# testing the net on the MNIST test data sets
print("\n" + "--test--")
test_error = 0
test_labels = mne.extractLabels("data/test_labels.bin")
print("There are {} test images available.".format(len(test_labels)))
custom_test_image_count = int(round(float(input("Enter the number of images you would like to load? > "))))
test_images = mne.extractImages("data/test_images.bin", custom_test_image_count)
for i in range(custom_test_image_count):
    test_error = nn.predict(test_labels[i], test_images[i], test_error)

print("error:", test_error)
