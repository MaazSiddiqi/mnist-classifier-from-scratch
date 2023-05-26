# MNIST Classifier from Scratch

A from-scratch implementation of a multi-layer, densely connected neural network that can classify handwritten images of numbers. Trained on the MNIST dataset with no external AI library dependencies!

After testing, the best accuracy this model produced was 95% (epochs=1000, learning_rate=0.01)!

## How to Run

1. Clone this repository
2. Run `python3 main.py` in the root directory of this repository

## How it Works

### TL;DR

The model is a multi-layer, densely connected neural network that uses the ReLU, Hyperbolic Tangent, and Sigmoid activation functions and the mean squared error loss function. The model is trained on the MNIST dataset, which contains 60,000 images of handwritten numbers (0-9) and their corresponding labels. The model is trained by feeding the images into the model, which then outputs a prediction. The loss function is then used to calculate the error between the prediction and the actual label. The error is then backpropagated through the model, and the weights are updated using gradient descent. This process is repeated for a specified number of epochs at a set learning rate.

### The Model

The implementation of this model is easily extensible to any number of layers and neurons per layer.

The model is initialized with a list of layers found in `layers.py`. Each layer is initialized with a number of input and output neurons. The activation functions for each layer are also specified in the list as another layer proceeding the layer it is activating, also found in `layers.py`. To see raw implementation of the activation functions, see `activations.py`.

The model also requires a loss function, which is used to calculate the error between the prediction and the actual label. The loss functions are specified in `loss.py`.

Initialize the network using the `Network` class found in `network.py`, and pass in the list of layers and the loss function.

For example, the following code creates a network with 3 layers, 784 input neurons, 128 neurons in the first hidden layer, 64 neurons in the second hidden layer, and 10 output neurons. The first hidden layer uses the ReLU activation function, the second hidden layer uses the Hyperbolic Tangent activation function, and the output layer uses the Sigmoid activation function. The loss function used is the mean squared error loss function. (This is the same model used in the `main.py` file.)

```python
from layers import Dense, ReLU_Activation, Softmax_Activation, Tanh_Activation
from loss import Mean_Squared_Error
from network import Network

# Create network
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
```

#### Training the Model

To train the model, call the `train` method on the network object. The `train` method takes in the training data, the training labels, the number of epochs to train for, and the learning rate.

```python
network.train(train_data, train_labels, epochs=1000, learning_rate=0.01)
```

You can also set `log=True` to print out the loss for each epoch.

```python
network.train(train_data, train_labels, epochs=1000, learning_rate=0.01, log=True)
```

#### Testing the Model

To test the model, call the `test` method on the network object. The `test` method takes in the testing data and the testing labels.

```python
accuracy = network.test(test_data, test_labels)
```
