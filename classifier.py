import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [ 2, 3, 0.5 ]

# Output of current layer
layer_outputs = []
for nueron_weights, bias in zip(weights, biases):
  output = np.dot(nueron_weights, inputs) + bias
  layer_outputs.append(output)

print(layer_outputs)
