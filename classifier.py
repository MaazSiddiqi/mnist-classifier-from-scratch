import numpy as np

inputs = [1, 2, 3, 2.5] # shape: 4,1

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
] # shape: 3,4

biases = [ 2, 3, 0.5 ] # shape: 3,1

output = np.dot(weights, inputs) + biases # shape: (3,4 * 4,1) + 3,1 = 3,1 + 3,1 = 3,1

print(output)
