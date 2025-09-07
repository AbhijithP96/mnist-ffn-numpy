# MNIST FFN - NumPy Neural Network with MLFLOW + FASTAPI

A from-scratch feedforward neural network implementation in Numpy, designed for MNIST dataset, with

- Flexible architecture (any number of layers, units, activations)
- Weight initialization strategies (uniform, heinit, xavier)
- Experiment tracking via MLflow
- REST API with FastAPI for training & inference
- Dockerized deployment for easy sharing

## 📂 Project Structure  

```bash
.
├── app.py                      # FastAPI app (train, load, predict endpoints)
├── train_exp.py                # CLI for training from JSON config
├── load_and_predict.py         # CLI for loading a model and inference
├── utils.py                    # Helper utilities
├── layers/                     # Linear layers
├── activations/                # Activation functions
├── loss/                       # Loss functions
├── optimizer/                  # Optimizers
├── trainer/                    # Training helpers
├── best_models/                # Saved models
├── tests/                      # Pytest unit tests
└── sample_test_images/         # Test images for prediction

```

## 🚀 Quick Start

### 1. Clone the repositiory
```bash
git clone https://github.com/AbhijithP96/mnist-ffn-numpy.git
cd mnist-ffn-numpy
```

### 2. Run with Docker

You have two options:

#### Option A: Pull a prebuilt image (recommended)
```
docker pull basilisk96/mnist-numpy-mlflow:latest
docker run -p 5000:5000 -p 8080:8080 basilisk96/mnist-numpy-mlflow:latest
```

#### Option B: Build locally (if you wish to modify code)
```
docker build -t {image_name} .
docker run -p 5000:5000 -p 8080:8080 {image_name}
```

👉 Replace {image_name} with your preferred image name

## ⚙️ Training

📌 Using CLI

```
python train_exp.py exp.json
```

Example ```exp.json```

```json
{
    "experiment_name": "Trial01",
    "experiments": {
        "1": {
            "run_name": "default_run",
            "layer_dict": [],
            "model_params": {
                "batch_size": 16,
                "epochs": 10,
                "learning_rate": 0.01
            }
        },
        "2": {
            "run_name": "4_layer_uniform_scale",
            "layer_dict": [
                {"input_dim": 784, "output_dim": 256, "activation": "sigmoid", "scale": "He-init"},
                {"input_dim": 256, "output_dim": 128, "activation": "relu", "scale": "Xavier"},
                {"input_dim": 128, "output_dim": 64,  "activation": "tanh", "scale": "uniform"},
                {"input_dim": 64,  "output_dim": 10,  "activation": "softmax", "scale": "uniform"}
            ],
            "model_params": {
                "batch_size": 16,
                "epochs": 10,
                "learning_rate": 0.01
            }
        }
    }    
}
```
⚠️ **Important Note:**  
The final layer **must use `softmax` activation**, since backpropagation is implemented assuming a `softmax + cross-entropy` combination.  
Using any other activation for the output layer may lead to incorrect gradients and failed training.  

### Tracking Runs

👉 All runs are tracked in MlFlow at ```http://0.0.0.0:8080```

## Loading Best Model & Inference

📌 Using CLI

```bash
python load_and_predict.py --exp {exp_name} --run {run_name}
```

👉 Replace {exp_name} with the name of your MLflow experiment, and {run_name} with the corresponding run you want to load.

Once the model is loaded successfully, you can run interactive inference:
```text
Give image path to run inference
/path/to/mnist_digit.png
Prediction: 7

Do you want to continue inference (y/N)
```

This will:

- Load the model artifacts saved under the chosen experiment/run.

- Accept any 28x28 grayscale image (PNG/JPG).

- Return the predicted digit in the console.


## 🛠️ Debug / Development

If you wish to modify the codes and run experiment locally.

- Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

- Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/mnist-ffn-numpy.git
cd mnist-ffn-numpy

# Install exact locked versions for reproducibility
uv sync --locked
```

- Run the app locally with:

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 5000
```


- After modifying the code, validate everything is working by running test:
```
uv run pytest
```

⚠️ **Important Note:** 
Rememeber to start mlflow server before running the app with:

```bash
mlflow server --host 0.0.0.0 --port 8080
```

## 🔌 API Endpoints

Once the FastAPI app is running (locally via uv run uvicorn app:app or in Docker), the following endpoints are available:

- ```POST /train```  ---> Train with JSON config (logs to MLFlow)

- ```GET /load_model``` ---> Load a trained model by experiment and run

- ```POST /predict ``` ---> Predict a digit from base64-encoded 28x28 grayscale image.

### Testing API

Using POSTMAN

#### 1. Open Postman and create a new request.
#### 2. Set the method and endpoint, for example:
```POST http://localhost:5000/train```

#### 3. In Body → form-data, add your JSON config file:
```text

Key: file
Type: File
Value: exp.json
```

#### 4. Send the request and inspect the JSON response.

#### 5. Similarly, test ```/load_model``` by providing query parameters.

👉 You can use ```predict.py``` script to test the ```/predict``` endpoint. The script will encode the image as base64 and send it to the /predict endpoint.
It will return the predicted digit directly in the console.







