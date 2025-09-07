from fastapi import FastAPI, UploadFile, File, Query
import json
from train import train_experiments
from utils import get_run_by_name, get_model_path, load_model, parse_input

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the MNIST FFN Training API"}

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    # Parse the incoming JSON request
    content = await file.read()
    data = json.loads(content)
    exp_name = data.get("experiment_name", "default_experiment")
    experiments = data['experiments']
    output = {}
    # get all experiments
    for key in experiments.keys():
        exp = experiments[key]
        run_name = exp.get("run_name", f"run_{key}")
        layer_dict = exp.get("layer_dict", [])
        model_params = exp.get("model_params", {})

        # Start the training process
        try:
            trainer = train_experiments(exp_name=exp_name, run_name=run_name, layer_dict=layer_dict, model_params=model_params)
            if trainer.training_done:
                test_acc = round(trainer.test_accuracy*100,2)
                output[f'exp_{key}'] = {"status": "success", "message": "Training completed successfully.", "test_data_accuracy": test_acc}
            else:
                output[f'exp_{key}'] = {"status": "error", "message": "Training did not complete successfully."}
        except Exception as e:
            return {"status": "runtime_error", "message": str(e)}

    return output

@app.get("/load_model")
async def load_model_endpoint(
    experiment_name: str = Query(..., description="Name of the experiment"),
    run_name: str = Query(..., description="Name of the run within the experiment")
):
    run = get_run_by_name(experiment_name, run_name)
    if run is None:
        return {"status": "error", "message": f"Run '{run_name}' not found in experiment '{experiment_name}'."}

    # get the model from local file system
    local_path = get_model_path(run.info.run_id)
    if local_path is None:
        return {"status": "error", "message": f"Model not found for run '{run_name}'."}

    app.state.model = load_model(local_path)
    print(app.state.model)
    return {"status": "success", "message": f"Model loaded successfully from run '{run_name}'."}

@app.post("/predict")
async def predict(data: dict):
    model = app.state.model
    if model is None:
        return {"status": "error", "message": "No model loaded. Please load a model first."}

    try:
        # Parse the incoming JSON request
        input_data = parse_input(data)

        # Perform prediction
        predictions = model(input_data)
        predicted_output = predictions.argmax(axis=0).tolist()

        return {"status": "success", "predictions": predicted_output}
    
    except Exception as e:
        return {"status" : "error", "message": str(e)}