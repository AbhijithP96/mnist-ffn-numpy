from data.mnist_loader import get_data, preprocess_data
from layers.linear import Linear
from model.neural_network import Model
from data.loader import Loader
from optmizer.optim import GradientDescent
from loss.entropy_loss import CrossEntropyLoss
from trainer.train import Trainer

import numpy as np
from tqdm import tqdm
import PIL.Image as Image



def main():

    # load data
    train_data, test_data, dev_data = get_data()
    print("Training data shape:", train_data[0].shape)
    print("Testing data shape:", test_data[0].shape)
    print("Development data shape:", dev_data[0].shape)

    # prrocess the data
    train_data, dev_data = preprocess_data(train_data, dev_data)

    # create data loader
    loader = Loader()
    train_loader = loader.get_batches(train_data[0], train_data[1], batch_size=16)
    dev_loader = loader.get_batches(dev_data[0], dev_data[1], batch_size=16)


    # input data dimenion
    input_dim = train_data[0].shape[1]

    # create layers
    layers = [
        Linear(input_dim=input_dim, output_dim=128, activation='relu', scale='He-init'),
        Linear(input_dim=128, output_dim=64, activation='relu', scale='He-init'),
        Linear(input_dim=64, output_dim=10, activation='softmax', scale='Xavier')
    ]


    # define the model
    model = Model(name='Trial04')
    model.add_layers(layers)

    model.summary()

    # define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = GradientDescent(model, learning_rate=0.01)

    # try a simple forward pass
    x_sample = np.reshape(train_data[0][0].T, (input_dim, 1))  # take first 5 samples
    print("Sample input shape:", x_sample.shape)
    output = model(x_sample)
    print("Output shape after forward pass:", output.shape)
    losses = []

    trainer = Trainer(model, 
                      optimizer,
                      loss_fn,
                      train_loader,
                      dev_loader,
                      epochs=10,
                      save_at=3,
                      mlflow_status=False)

    trainer.train(experiment_name='mnist_trial_01')
    #trainer.plot_metrics()
    #trainer.save(path='./params', filename='mnist_model.pkl')

    #test(test_data, best_model_path='./params/mnist_model.pkl', model=model)

def test(test_loader, best_model_path, model):
    
    import pickle
    import matplotlib.pyplot as plt
    with open(best_model_path, 'rb') as f:
        best_params = pickle.load(f)
    
    # load the best parameters into the model
    for i, layer in enumerate(model.layers):
        layer.weights = best_params[f'layer{i}']['weights']
        layer.biases = best_params[f'layer{i}']['biases']

    print(len(test_loader[0]))


    for i in range(len(test_loader[0])):
        print(i)

        if i > 5:
            break

        image = test_loader[0][i]
        label = test_loader[1][i]

        X = image.copy().astype('float32') / 255.0
        X = X.reshape(1, X.shape[0]*X.shape[1])

        output = model(X.T)
        y_pred = np.argmax(output, axis=0)

        plt.imshow(np.uint8(image))
        plt.title(f'True:{label} Pred:{y_pred}')
        plt.axis('off')
        plt.show()
        


if __name__ == "__main__":
    main()