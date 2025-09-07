import pytest
import numpy as np

from activations.activations import Forward

np.random.seed(42)

def test_activation_relu():
    """Test ReLU activation function."""
    x = np.array([[-1, 2, -3], [4, -5, 6]])
    output = Forward.relu(x)
    expected_output = np.array([[0, 2, 0], [4, 0, 6]])
    assert np.array_equal(output, expected_output), "ReLU activation failed."

def test_activation_sigmoid():
    """Test Sigmoid activation function."""
    x = np.array([[0, 0], [0, 0]])
    output = Forward.sigmoid(x)
    expected_output = [[0.5, 0.5], [0.5, 0.5]]
    assert np.allclose(output, expected_output), "Sigmoid activation failed."

def test_activation_tanh():
    """Test Tanh activation function."""
    x = np.array([[0, 0], [0, 0]])
    output = Forward.tanh(x)
    expected_output = [[0, 0], [0, 0]]
    assert np.allclose(output, expected_output), "Tanh activation failed."

def test_activation_softmax():
    """Test Softmax activation function."""
    x = np.array([[1, 2, 3],
                  [1, 2, 3]])           # shape (2 units, 3 samples)
    output = Forward.softmax(x)

    # Sum over classes (axis=0) -> should be ones for each sample
    expected_output = np.sum(output, axis=0)
    np.testing.assert_allclose(expected_output,np.ones(output.shape[1]),rtol=1e-6,err_msg="Softmax activation failed.")
    assert np.all(output >= 0) and np.all(output <= 1), "Softmax output not in [0, 1] range."