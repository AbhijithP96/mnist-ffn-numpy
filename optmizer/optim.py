import numpy as np
from model.neural_network import Model

class GradientDescent:
    def __init__(self, model: Model, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.model = model

    def update(self):
        """Update parameters using gradient descent.
        Args:
            parameters (dict): Dictionary of model parameters.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        for key in self.model.parameters.keys():
            self.model.parameters[key]['weights'] -= self.learning_rate * self.model.grads[key]['weights']
            self.model.parameters[key]['biases'] -= self.learning_rate * self.model.grads[key]['biases']
        return self.model.parameters