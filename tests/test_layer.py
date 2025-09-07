import pytest
import numpy as np

from layers.linear import Linear

np.random.seed(42)

def test_linear_layer_initialization():
    input_dim = 5
    output_dim = 3
    layer = Linear(input_dim=input_dim, output_dim=output_dim, activation='relu')
    
    assert layer.input_dim == input_dim, "Input dimension not set correctly."
    assert layer.output_dim == output_dim, "Output dimension not set correctly."
    assert layer.weights.shape == (output_dim, input_dim), "Weights shape is incorrect."
    assert layer.biases.shape == (output_dim, 1), "Biases shape is incorrect."
    assert callable(layer.activation), "Activation function is not callable."

def test_linear_layer_forward_pass_shape():
    input_dim = 4
    output_dim = 2
    num_samples = 3
    layer = Linear(input_dim=input_dim, output_dim=output_dim, activation='relu')
    
    # Create a sample input
    x = np.random.randn(input_dim, num_samples)
    
    # Perform forward pass
    output = layer.forward(x)
    
    assert output.shape == (output_dim, num_samples), "Output shape is incorrect."
    

def test_linear_layer_forward_pass_values():
    input_dim = 1
    output_dim = 1
    layer = Linear(input_dim=input_dim, output_dim=output_dim, activation='linear')

    # Manually set weights and biases for predictable output
    layer.weights = np.array([[2]])
    layer.biases = np.array([[3]])

    # Create a sample input
    x = np.array([[1]])

    # Perform forward pass
    output = layer.forward(x)

    # Check if the output is as expected
    expected_output = 2 * 1 + 3
    assert np.all(output == expected_output), "Forward pass output values are incorrect."

def test_linear_layer_no_activation():
    input_dim = 3
    output_dim = 2
    layer = Linear(input_dim=input_dim, output_dim=output_dim, activation=None)

    assert layer.activate is not None, "Activation function should default to linear."

def test_linear_layer_zero_input_bias():
    input_dim = 3
    output_dim = 2
    num_samples = 4
    layer = Linear(input_dim=input_dim, output_dim=output_dim, activation='linear')
    
    # Create a zero input
    x = np.zeros((input_dim, num_samples))
    
    # Perform forward pass
    output = layer.forward(x)
    
    # Since input is zero, output should be equal to biases
    assert np.all(output == layer.biases), "Output should equal biases when input is zero."

def test_linear_backward_shapes():
    
    layer = Linear(input_dim=4, output_dim=3, activation='relu')
    x = np.random.randn(4, 2)
    out = layer.forward(x)
    dout = np.random.randn(3, 2)  # upstream gradient
    dx, dw, db = layer.backward(dout)

    # Check gradient shapes
    assert dx.shape == (4, 2), "Input gradient shape mismatch"
    assert dw.shape == (3, 4), "Weight gradient shape mismatch"
    assert db.shape == (3, 1), "Bias gradient shape mismatch"

