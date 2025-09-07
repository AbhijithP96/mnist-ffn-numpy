import pytest
import numpy as np
from layers.linear import Linear
from model.neural_network import Model
from loss.entropy_loss import CrossEntropyLoss
from optmizer.optim import GradientDescent
from trainer.train import Trainer

np.random.seed(42)

def test_loss_decrease_during_training():
    # Create a sample dataset
    X = np.random.randn(28*28, 10) # 10 samples of 28x28 images
    y = np.eye(10)[np.random.randint(0, 10, size=(10,))].T  # One-hot encoded labels for 10 classes

    # Create a simple model
    layers = [
        Linear(input_dim=28*28, output_dim=64, activation='relu', scale='He-init'),
        Linear(input_dim=64, output_dim=10, activation='softmax', scale='Xavier')
    ]

    model = Model(name='TestModel')
    model.add_layers(layers)

    loss_fn = CrossEntropyLoss()
    optimizer = GradientDescent(model, learning_rate=0.01)

    logits = model(X)
    initial_loss = loss_fn(logits, y)

    # Perform a training step
    trainer = Trainer(model, 
                      optimizer,
                      loss_fn,
                      train_loader=[(X.T, y.T)],  # Single batch
                      val_loader=[(X.T, y.T)],  # Single batch
                      epochs=1,
                      mlflow_status=False)
    trainer.train_one_epoch()
    logits_after = model(X)
    final_loss = loss_fn(logits_after, y)
    assert final_loss < initial_loss, "Loss did not decrease after training step."
