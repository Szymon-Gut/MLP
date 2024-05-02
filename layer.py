import numpy as np
from activation_functions import Sigmoid

class Layer:
    """
    A class representing a single layer in a neural network.

    This class encapsulates the properties and methods of a neural network layer, including weight initialization,
    forward propagation, activation, and weight update.

    Attributes:
        shape (tuple): The shape of the layer's weight matrix, specifying the number of input and output neurons.
        activation (Activation, optional): The activation function applied to the layer's output. Defaults to Sigmoid().

    Examples:
        # Create a layer with 100 input neurons and 50 output neurons using the Sigmoid activation function.
        >>> layer = Layer(shape=(100, 50), activation=Sigmoid())

        # Perform forward propagation and activation on input data.
        >>> weighted_input = layer.calculate(input_data)
        >>> output = layer.activate(weighted_input)

        # Update layer weights and biases based on delta values.
        >>> layer.update(delta_weights, delta_biases)
    """

    def __init__(self, shape, activation=Sigmoid()):
        """
        Initializes a new layer with the specified shape and activation function.

        Args:
            shape (tuple): The shape of the layer's weight matrix, specifying the number of input and output neurons.
            activation (Activation, optional): The activation function applied to the layer's output. Defaults to Sigmoid().
        """
        self.shape = shape
        self.activation = activation
        self._initialize_weights()
        self.weighted_input = None
        self.output = None

    def _initialize_weights(self, min_val=-1, max_val=1):
        """
        Initializes the layer's weights and biases with random values within a specified range.

        Args:
            min_val (float, optional): The minimum value for weight initialization. Defaults to -1.
            max_val (float, optional): The maximum value for weight initialization. Defaults to 1.
        """
        self.weights = np.random.uniform(min_val, max_val, size=self.shape)
        self.biases = np.random.uniform(min_val, max_val, size=(self.shape[1], 1))

    def calculate(self, x):
        """
        Calculates the weighted input of the layer.

        Args:
            x (array): The input data to the layer.

        Returns:
            array: The weighted input of the layer.
        """
        self.weighted_input = (self.weights.T @ x) + self.biases
        return self.weighted_input

    def activate(self, x):
        """
        Applies the activation function to the input data.

        Args:
            x (array): The input data to the layer.

        Returns:
            array: The activated output of the layer.
        """
        self.output = self.activation.calculate(x)
        return self.output

    def update(self, delta_weights, delta_biases):
        """
        Updates the layer's weights and biases based on the provided delta values.

        Args:
            delta_weights (array): The delta values for updating the weights.
            delta_biases (array): The delta values for updating the biases.
        """
        self.weights += np.array(np.reshape(delta_weights, self.shape))
        self.biases += np.array(np.reshape(delta_biases, (self.shape[1], 1)))
