import numpy as np

class Forward:

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Compute the ReLU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying ReLU.
        """
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute the sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying sigmoid.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Compute the tanh activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying tanh.
        """
        return np.tanh(x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Compute the softmax activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying softmax.
        """
        # shift for numerical stability
        x_shifted = x - np.max(x, axis=0, keepdims=True)
        e_x = np.exp(x_shifted)
        return e_x / e_x.sum(axis=0, keepdims=True)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear activation function (identity function).

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array (same as input).
        """
        return x
    

class Backward:
    
    @staticmethod
    def softmax_backward( *args) -> np.ndarray:
        """Compute the gradient of the loss with respect to the input of the softmax layer.
            
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the softmax layer.
        """
        # cross-entropy loss combined with softmax
        return args[0] - args[1]

    @staticmethod
    def relu_backward(*args) -> np.ndarray:
        """Compute the gradient of the loss with respect to the input of the ReLU layer.
            
        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the ReLU layer.
        """
        return args[0] * (args[1] > 0)

    @staticmethod
    def sigmoid_backward(*args) -> np.ndarray:
        """Compute the gradient of the loss with respect to the input of the sigmoid layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the sigmoid layer.
        """
        sigmoid_input = args[2]
        return args[0] * sigmoid_input * (1 - sigmoid_input)

    @staticmethod
    def tanh_backward(*args) -> np.ndarray:
        """Compute the gradient of the loss with respect to the input of the tanh layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the tanh layer.
        """
        tanh_input = args[2]
        return args[0] * (1 - tanh_input ** 2)

    @staticmethod
    def linear_backward(*args) -> np.ndarray:
        """Compute the gradient of the loss with respect to the input of the linear layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the linear layer.
        """
        return args[0] # identity