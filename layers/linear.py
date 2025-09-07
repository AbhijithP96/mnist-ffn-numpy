import numpy as np
from activations.activations import Forward, Backward

ACTIVATIONS = {
    'relu': Forward.relu,
    'sigmoid': Forward.sigmoid,
    'tanh': Forward.tanh,
    'linear': Forward.linear,
    'softmax': Forward.softmax
}

ACTIVATIONS_DERIVATIVES = {
    'relu': Backward.relu_backward,
    'sigmoid': Backward.sigmoid_backward,
    'tanh': Backward.tanh_backward,
    'linear': Backward.linear_backward,
    'softmax': Backward.softmax_backward
}

SCALE = {
    'He-init': lambda x: np.sqrt(2. / x),
    'Xavier': lambda x: 1. / np.sqrt(x),
    'uniform': lambda x: 0.01
}

class Linear:

    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu', scale= 'uniform'):
        """Initialize a linear layer with weights, biases, and activation function.
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            activation (str): Activation function to use ('relu', 'sigmoid', 'softmax').
            scale (str): Scale to initialize the weights of the layer.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activate = activation.lower() if activation else 'linear'
        self.activation = ACTIVATIONS.get(self.activate)
        self.activation_derivative = ACTIVATIONS_DERIVATIVES.get(self.activate)
        self.layer_scale = scale
        # Initialize weights and biases
        self.weights = np.random.randn(output_dim, input_dim) * SCALE.get(scale)(input_dim)
        self.biases = np.zeros((output_dim, 1))
        self.caches = []  # To store intermediate values for backpropagation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass through the layer.
        Args:
            x (np.ndarray): Input data of shape (input_dim, num_samples).

        Returns:
            np.ndarray: Output data of shape (output_dim, num_samples) after applying the linear transformation and activation function.
        """
        z = np.dot(self.weights, x) + self.biases
        a = self.activation(z)
        self.caches.append((x, z, a))  # Store input, linear output, activated output
        return a
    
    def backward(self, dout: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Perform the backward pass through the layer.
        Args:
            dout (np.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        x, z, a = self.caches.pop()
        m = x.shape[1]

        if self.activate == 'softmax':
            # only valid if loss is cross-entropy
            dz = self.activation_derivative(a, y)
        else:
            dz = self.activation_derivative(dout, z, a)

        dW = (dz @ x.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dx = self.weights.T @ dz

        return dx, dW, db
    
   