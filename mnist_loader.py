#
# Kaggle reference that demonstrates how to read "MNIST Dataset"
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
#
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)

    def process_data(self, X, Y, limit=None):
        X = np.array(X)
        Y = np.array(Y)

        # Reshape X to (n, 784)
        X = X.reshape(X.shape[0], 784)

        # Normalize X
        X = X.astype("float32")
        X /= 255

        # Convert Y to one-hot encoding of size 10 (number of classes)
        # and make it column vector as expected by network
        Y = np.eye(10)[Y]
        Y = Y.reshape(Y.shape[0], 10)

        # Limit data if needed
        if limit is not None:
            X = X[:limit]
            Y = Y[:limit]

        return X, Y
