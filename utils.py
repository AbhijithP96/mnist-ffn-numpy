from mlflow.tracking import MlflowClient
import pickle
import numpy as np
import base64
from PIL import Image
from io import BytesIO

from layers.linear import Linear
from model.neural_network import Model

client = MlflowClient(tracking_uri='http://0.0.0.0:8080')

def get_run_by_name(experiment_name, run_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    for run in runs:
        if run.data.tags.get('mlflow.runName') == run_name:
            return run
    return None

def get_model_path(run_id):
    artifacts = client.list_artifacts(run_id)
    for artifact in artifacts:
        if artifact.path.endswith('.pkl'):
            return artifact.path
    return None

def load_model(local_path):
    with open(f'./best_models/{local_path}', 'rb') as f:
        params = pickle.load(f)
    
    num_layers = len(params.keys())
    model_layers = []

    for i in range(num_layers):
        layer_params = params[f'layer{i}']

        # load weights, biases, activation
        weights = layer_params['weights']
        biases = layer_params['biases']
        activation = layer_params['activation']
        scale = layer_params.get('scale', 'uniform')

        # get input and output dimensions
        input_dim = weights.shape[1]
        output_dim = weights.shape[0]

        # create layer and set parameters
        layer = Linear(input_dim=input_dim, output_dim=output_dim, activation=activation, scale=scale)
        layer.weights = weights
        layer.biases = biases

        # append to model layers
        model_layers.append(layer)

    # create model and add layers
    model = Model(name='Best Model')
    model.add_layers(model_layers)
    return model

def parse_input(input_data: dict):
    # decode base64 image
    img_data = input_data.get('image')
    if img_data is None:
        raise ValueError("Input data must contain 'image' key with base64 encoded image string.")
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes)).convert('L')
    img = img.resize((28,28))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_array = img_array.reshape(-1, 1)  # Reshape to (784, 1) for MNIST
    return img_array