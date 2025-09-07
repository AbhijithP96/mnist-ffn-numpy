
class Model:
    def __init__(self, name='NeuralNetwork'):
        self.name = name
        self.layers = None
        self.parameters = {}
        self.grads = {}
        self.train = True
        self.loss = []  # To store loss values during training
        self.accuracy = []  # To store accuracy values during training
        self.eval = False  # To track if the model is in evaluation mode

    def add_layers(self, layers: list):
        """Add layers to the neural network model.
        Args:
            layers (list): List of layer objects to be added to the model.
        """
        self.layers = layers
        for i, layer in enumerate(layers):
            self.parameters[f'layer{i}'] = {
                'weights': layer.weights,
                'biases': layer.biases,
                'activation' : layer.activate,
                'scale' : layer.layer_scale
            }

    def __call__(self, x):
        """Perform a forward pass through the network.
        Args:
            x (np.ndarray): Input data of shape (input_dim, num_samples).

        Returns:
            np.ndarray: Output data after passing through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def print_summary(self):
        """Print a summary of the model architecture."""
        print(f'Model Summary for {self.name}')
        print('-----------------------------------')
        for i, layer in enumerate(self.layers):
            print(f'Layer {i + 1}: {layer.__class__.__name__}')
            print(f'  Input Dim: {layer.input_dim}')
            print(f'  Output Dim: {layer.output_dim}')
            print(f'  Weights Shape: {layer.weights.shape}')
            print(f'  Biases Shape: {layer.biases.shape}')
            print(f'  Activation: {layer.activate}')
            print('-----------------------------------')

    def get_summary(self):
        """Get a summary of the model architecture."""
        summary = {
            'Model Name': self.name,
            'Layers': []
        }
        for i, layer in enumerate(self.layers):
            layer_name = 'Layer ' + str(i + 1)
            layer_info = {layer_name: {}}
            layer_info[layer_name] = {
                'Type': layer.__class__.__name__,
                'Input Dim': layer.input_dim,
                'Output Dim': layer.output_dim,
                'Weights Shape': layer.weights.shape,
                'Biases Shape': layer.biases.shape,
                'Activation': layer.activate
            }
            summary['Layers'].append(layer_info)
        
        return summary
    
    def backward(self, y=None):
        """Perform backward pass through the network.
        Args:
            y (np.ndarray): True labels for the input data, required for loss computation.

        Returns:
            None
        """
        dout = None
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                dout, dw, db = layer.backward(dout, y)  # output layer
            else:
                dout, dw, db = layer.backward(dout)     # hidden layers
            self.grads[f'layer{len(self.layers) - 1 - i}'] = {'weights': dw, 'biases': db}


    