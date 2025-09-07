import base64
import json
import requests
import argparse

def load_best_model(exp_name, run_name):
    url = f'http://0.0.0.0:5000/load_model?experiment_name={exp_name}&run_name={run_name}'

    response = requests.get(url)
    return response

def predict(image_path):

    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # send to the prediction endpoint
    url = 'http://0.0.0.0:5000/predict'
    payload = {"image": encoded_image}

    response = requests.post(url, json=payload)
    print(response.json())


def get_args():
    parser = argparse.ArgumentParser(description='Load a trained model from MLflow and run predictions.')

    parser.add_argument('--exp', type=str, default='Trial01' ,help='Name of the MLflow experiment (default: Trial01)')
    parser.add_argument('--run', type=str, default='default_run', help='Name of the run within the experiment to load (default: default_run).') 

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    exp = args.exp
    run = args.run

    res = load_best_model(exp, run)

    if res.status_code == 200:
        print(res)

        while True:
            print('Give image path to run inference')
            img_path = input()
            predict(image_path=img_path)

            print('Do you want to continue inference (y/N)')
            option = input()

            if option.lower() == 'n':
                break

    else:
        print(f"Failed to load model. Server response: {res.status_code} {res.text}")

