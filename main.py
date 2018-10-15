from neural_net import MnistNet
from mnist_extractor1 import MNISTExtractor

mne = MNISTExtractor()
foldername = "net_data"

# print headline and options
print("\n" + """          ---Welcome to---
A NEURAL NET RECOGNISING HANDWRITTEN DIGITS
    ---based on the MNIST dataset--- 
""")

print("""
1) Train a new neural net
2) Load an already existing neural net file
""")

while True:
    choice = input("Please enter your choice > ")

    # Train a new neural net
    if choice == "1":
        # define base parameters of the net
        print("\n" + "Your net concists of 784 input nodes (28 * 28 pixel image) and 10 output nodes (10 numbers).")
        number_hidden_nodes = int(round(float(input("How many hidden layers would you like to use? > "))))
        layer_size = [784]
        for i in range(number_hidden_nodes):
            layer_size.append(int(round(float(input("Enter the number of hidden nodes for the " + str(i + 1) +
                                                    ". layer! > ")))))
        layer_size.append(10)
        learning_rate = float(input("Enter the learning rate! > "))

        nn = MnistNet(hidden_layer_size=layer_size, learning_rate=learning_rate)
        nn.generate_weights()

        # Loading the MNIST training data set
        print("\n" + "Loading MNIST training data labels...")
        train_labels = mne.extractLabels("mnist_data/train_labels.bin")
        custom_image_count = nn.enter_image_count(len(train_labels))

        print("Loading {} MNIST training images...".format(custom_image_count))
        train_images = mne.extractImages("mnist_data/train_images.bin", custom_image_count + 1)

        # -----TRAINING-----
        epochs = int(round(float(input("How many epochs should be executed? (Training will stop automaticaly after 7 "
                                       "epochs without improvement)> "))))

        print("\n" + "Executing neural_net")
        print("This could take some time...")

        for epoch in range(epochs):
            for i in range(custom_image_count - 1):
                nn.backpropagation(train_labels[i], nn.execute(train_images[i]))
            train_error, accuracy  = nn.calculate_accuracy(train_labels, train_images, custom_image_count)
            print("Epoch {}: {} error: {} {} accuracy: {}%".format(epoch + 1, " " * abs(7 - len(str(epoch + 1))),
                                                    train_error, " " * abs((7 - len(str(train_error)))), accuracy))
            if not nn.cancel_training():
                break

        # -----TESTING-----
        print("\n" + "Testing the trained neural net on the MNIST test data set")
        test_error = 0
        test_labels = mne.extractLabels("mnist_data/test_labels.bin")
        custom_test_image_count = nn.enter_image_count(len(test_labels))
        test_images = mne.extractImages("mnist_data/test_images.bin", custom_test_image_count)
        test_error, test_accuracy = nn.calculate_accuracy(test_labels, test_images, custom_test_image_count)
        print("error: {}    accuracy: {}".format(test_error, test_accuracy))

        # -----SAVING-----
        # save scores
        while True:
            user_in_save_scores = input("\n" + "Would you like to save the score of your neural net? (y/n) > ")
            if user_in_save_scores == "y":
                nn.save_json_results(foldername, custom_image_count, epochs, layer_size, test_accuracy)
                print("The scores are accessable by starting table.py in python and then visiting"
                      "'http://localhost:4000/'")

                # save net
                user_in_save_net = input("\n" + "Would you also like to save the trained neural net (This allows a "
                                                "faster usage, without having to train the net again)? (y/n) > ")
                if user_in_save_net == "y":
                    nn.save(foldername)
                    break
                elif user_in_save_net == "n":
                    break
                else:
                    print("\033[2;31;40m" + "Error: Invalid input! Enter either 'y' to save your net "
                                            "or 'n' to quit" + "\033[0m")

            elif user_in_save_scores == "n":
                break
            else:
                print("\033[2;31;40m" + "Error: Invalid input! Enter either 'y' to continue or 'n' to quit" + "\033[0m")
        break

    # Load an existing neural net
    elif choice == "2":
        nn = MnistNet()
        filename = foldername + "/" + input("Enter the name of your file (e.g.: 'net3') > ")
        nn.load(filename)

        # -----TESTING-----
        print("\n" + "Testing the trained neural net on the MNIST test data set")
        test_error = 0
        test_labels = mne.extractLabels("mnist_data/test_labels.bin")
        custom_test_image_count = nn.enter_image_count(len(test_labels))
        test_images = mne.extractImages("mnist_data/test_images.bin", custom_test_image_count)
        test_error, test_accuracy = nn.calculate_accuracy(test_labels, test_images, custom_test_image_count)
        print("error: {}    accuracy: {}".format(test_error, test_accuracy))
        break

    else:
        print("Error: Incorrect Input: Enter either '1' or '2'!")
