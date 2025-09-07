import numpy as np
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import mlflow

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, epochs, save_at=None, mlflow_status=False, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.losses = {
            'train_losses' : [],
            'val_losses' : []
        }
        self.acc = {
            'train_acc' : [],
            'val_acc' : []
        }

        self.save_at = save_at
        self.training_done = False
        self.metrics = None

        self.mlflow_on = mlflow_status
        self.best_acc = 0.0  # To track the best validation accuracy
        self.best_model_params = None  # To store the best model parameters
        self.best_epoch = -1  # To track the epoch of the best model
        self.model_summary = self.model.get_summary()
        self.exp_name = kwargs.get('exp_name', 'MNIST_FFN_Numpy')
        self.batch_size = kwargs.get('batch_size', 16)

    def train_one_epoch(self):

        for data in tqdm(self.train_loader, desc='Training'):
            X, y = data

            # output from the model
            output = self.model(X.T)

            # compute the loss
            loss = self.loss_fn(y.T, output)

            # backprop and update the parameters
            self.model.backward(y.T)
            self.optimizer.update()

            self.losses['train_losses'].append(loss)
            acc = np.mean(np.argmax(output, axis=0) == np.argmax(y.T, axis=0))
            self.acc['train_acc'].append(acc)

    def test_one_epoch(self):

        for data in tqdm(self.val_loader, desc='Validation'):
            X, y = data
            # output from the model
            output = self.model(X.T)

            # comput the loss
            loss = self.loss_fn(y.T, output)

            self.losses['val_losses'].append(loss)
            acc = np.mean(np.argmax(output, axis=0) == np.argmax(y.T, axis=0))
            self.acc['val_acc'].append(acc)

    def train(self, experiment_name=None):
        if self.mlflow_on:
            self.train_with_mlflow(experiment_name)
        else:
            self.train_without_mlflow()

    def train_without_mlflow(self):

        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []

        for epoch in range(self.epochs):
            print('---------------------------------------------------')
            print(f"Epoch {epoch + 1}")

            # train the model
            self.train_one_epoch()

            # test the model
            self.test_one_epoch()

            # print the losses
            train_loss = np.mean(self.losses['train_losses'])
            val_loss = np.mean(self.losses['val_losses'])
            print(f'Training Loss : {train_loss}' )
            print(f'Validation Loss: {val_loss}')

            self.losses['train_losses'] = []
            self.losses['val_losses'] = []

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_acc.append(np.mean(self.acc['train_acc']))
            val_acc.append(np.mean(self.acc['val_acc']))

            self.check_best_model(val_acc[-1], train_acc[-1])

        self.training_done = True
        self.metrics = {
            'train_losses' : train_losses,
            'val_losses' : val_losses,
            'train_acc' : train_acc,
            'val_acc' : val_acc
        }

    def train_with_mlflow(self, run_name=None):
    

        mlflow.set_tracking_uri(uri='http://0.0.0.0:8080')
        mlflow.set_experiment(self.exp_name)

        with mlflow.start_run(run_name=run_name) as run:

            self.run_id = run.info.run_id

            mlflow.log_param('optimizer', self.optimizer.__class__.__name__)
            mlflow.log_param('learning_rate', self.optimizer.learning_rate)
            mlflow.log_param('epochs', self.epochs)
            mlflow.log_param('batch_size', self.batch_size)
            mlflow.log_param('loss_function', self.loss_fn.__class__.__name__)

            train_losses = []
            val_losses = []
            train_acc = []
            val_acc = []

            for epoch in range(self.epochs):
                print('---------------------------------------------------')
                print(f"Epoch {epoch + 1}")

                # train the model
                self.train_one_epoch()

                # test the model
                self.test_one_epoch()

                # print the losses
                train_loss = np.mean(self.losses['train_losses'])
                val_loss = np.mean(self.losses['val_losses'])
                print(f'Training Loss : {train_loss}' )
                print(f'Validation Loss: {val_loss}')

                mlflow.log_metric('train_loss', train_loss, step=epoch)
                mlflow.log_metric('val_loss', val_loss, step=epoch)

                self.losses['train_losses'] = []
                self.losses['val_losses'] = []

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                train_acc.append(np.mean(self.acc['train_acc']))
                val_acc.append(np.mean(self.acc['val_acc']))

                mlflow.log_metric('train_acc', train_acc[-1], step=epoch)
                mlflow.log_metric('val_acc', val_acc[-1], step=epoch)

                self.check_best_model(val_acc[-1], train_acc[-1], epoch)

            self.training_done = True
            self.metrics = {
                'train_losses' : train_losses,
                'val_losses' : val_losses,
                'train_acc' : train_acc,
                'val_acc' : val_acc
            }

            mlflow.log_dict(self.model_summary, 'model_summary.json')
            mlflow.log_dict(self.metrics, 'metrics.json')

            # save the best model parameters
            if self.best_model_params:
                name = run_name if run_name else 'default_experiment'
                filename = f'model_{name}.pkl'
                param_path = self.save(filename=filename)
                mlflow.log_artifact(param_path)

    def plot_metrics(self):
        """Plot training and testing loss and accuracy curves.
        """
        if not self.training_done:
            raise ValueError("Training not completed. Cannot plot metrics.")

        epochs = range(1, self.epochs + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Loss
        ax1.plot(epochs, self.metrics['train_losses'], label='Training Loss')
        ax1.plot(epochs, self.metrics['val_losses'], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Plot Accuracy
        ax2.plot(epochs, self.metrics['train_acc'], label='Training Accuracy')
        ax2.plot(epochs, self.metrics['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()


    def check_best_model(self, val_acc, train_acc, epoch=None):
        """Check and update the best model parameters based on validation accuracy.
        Args:
            val_acc (float): Current validation accuracy.
            train_acc (float): Current training accuracy.
        """
        if val_acc > train_acc and val_acc > self.best_acc:

            self.best_acc = val_acc
            self.best_model_params = self.model.parameters.copy()
            self.best_epoch = epoch if epoch is not None else -1

    def save(self, path='./best_models', filename='model_params.pkl'):
        """Save the model parameters to a file.
        Args:
            path (str): Directory to save the parameters.
            filename (str): Name of the file to save the parameters.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        full_path = os.path.join(path, filename)
        
        # save the best model parameters if available
        params_to_save = self.best_model_params if self.best_model_params else self.model.parameters

        with open(full_path, 'wb') as f:
            pickle.dump(params_to_save, f)
        
        return full_path
    
    def test(self, test_loader):

        test_acc = []
        self.model.parameters = self.best_model_params

        for data in tqdm(test_loader, desc='Testing'):
            X, y = data

            # output from the model
            output = self.model(X.T)

            # get the accuracy
            acc = np.mean(np.argmax(output, axis=0) == np.argmax(y.T, axis=0))
            test_acc.append(acc)

        # set the mean accuracy on the test data
        self.test_accuracy = np.mean(test_acc)
        client = mlflow.tracking.MlflowClient(tracking_uri='http://0.0.0.0:8080')
        client.log_metric(self.run_id, 'test_accuracy', self.test_accuracy)