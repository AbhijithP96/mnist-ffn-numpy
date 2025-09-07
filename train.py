from data.mnist_loader import get_data, preprocess_data
from layers.linear import Linear
from model.neural_network import Model
from data.loader import Loader
from optmizer.optim import GradientDescent
from loss.entropy_loss import CrossEntropyLoss
from trainer.train import Trainer

import numpy as np
from tqdm import tqdm

def default_model():

    layers = [
        Linear(input_dim=28*28, output_dim=128, activation='relu', scale='He-init'),
        Linear(input_dim=128, output_dim=64, activation='relu', scale='He-init'),
        Linear(input_dim=64, output_dim=10, activation='softmax', scale='Xavier')
    ]

    model = Model(name='DefaultModel')
    model.add_layers(layers)

    return model

def create_new_model(layer_dict, model_name='CustomModel'):

    layers = []
    input_dim = 28*28  # Default input dimension for MNIST

    for layer in layer_dict:
        if 'input_dim' not in layer:
            layer['input_dim'] = input_dim
        layers.append(Linear(**layer))
        input_dim = layer['output_dim']  # Update input_dim for the next layer

    model = Model(name=model_name)
    model.add_layers(layers)
    
    return model

def train_experiments(exp_name, run_name, layer_dict, model_params):
    # load data
    train_data, test_data, dev_data = get_data()

    # prrocess the data
    train_data, dev_data, test_data = preprocess_data(train_data, dev_data, test_data)

    # get batch size, epochs, learning rate from model_params
    batch_size = model_params.get('batch_size', 16)
    epochs = model_params.get('epochs', 10)
    learning_rate = model_params.get('learning_rate', 0.01)

    # create data loader
    loader = Loader()
    train_loader = loader.get_batches(train_data[0], train_data[1], batch_size=batch_size)
    dev_loader = loader.get_batches(dev_data[0], dev_data[1], batch_size=batch_size)
    test_loader = loader.get_batches(test_data[0], test_data[1], batch_size=32)

    if not layer_dict:
        model = default_model()
    else:
        model = create_new_model(layer_dict, model_name=run_name)

    # define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = GradientDescent(model, learning_rate=learning_rate)

    trainer = Trainer(model, 
                      optimizer,
                      loss_fn,
                      train_loader=train_loader,
                      val_loader=dev_loader,
                      epochs=epochs,
                      mlflow_status=True,
                      exp_name=exp_name,
                      batch_size=batch_size)
    
    trainer.train(experiment_name=run_name)
    trainer.test(test_loader)
    
    return trainer

if __name__ == "__main__":
    run_name = "default_run"
    layer_dict = []
    model_params = {
        "batch_size": 16,
        "epochs": 10,
        "learning_rate": 0.01
    }
    trainer = train_experiments(run_name, layer_dict, model_params)
    if trainer.training_done:
        print("Training completed successfully.")
        print("Metrics:", trainer.metrics)
    else:
        print("Training did not complete successfully.")