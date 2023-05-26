#
# Verify Reading Dataset via MnistDataloader class
#
from os.path import join
from mnist_loader import MnistDataloader
from layers import Dense, ReLU_Activation, Softmax_Activation, Tanh_Activation
from network import Network
from loss import Mean_Squared_Error
import numpy as np

np.seterr(all="raise")

#
# Set file paths based on added MNIST Datasets
#
input_path = "./dataset"
training_images_filepath = join(
    input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = join(
    input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")


#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)

# Load training and test data
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# Process data to correct shapes for network
x_train, y_train = mnist_dataloader.process_data(x_train, y_train, 1000)
x_test, y_test = mnist_dataloader.process_data(x_test, y_test, 20)

# # Create network
# network_layers = [
#     Dense(784, 128),
#     ReLU_Activation(),
#     Dense(128, 64),
#     ReLU_Activation(),
#     Dense(64, 10),
#     Softmax_Activation(),
# ]

network_layers = [
    Dense(784, 128, initializer_scale=2),
    ReLU_Activation(),
    Dense(128, 64),
    Tanh_Activation(),
    Dense(64, 10),
    Softmax_Activation(),
]
mse = Mean_Squared_Error()
network = Network(network_layers, mse)

# error = network.train(x_train[:1], y_train[:1], epochs=2, learning_rate=0.01, log=True)
error = network.train(x_train, y_train, epochs=500, learning_rate=0.01, log=True)
accuracy = network.test(x_test, y_test)

print(f"Accuracy: {accuracy}")
