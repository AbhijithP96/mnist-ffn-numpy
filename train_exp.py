import requests
import json
import argparse

def train_experiments(exp_json_path):

    url = 'http://localhost:5000/train'

    with open(exp_json_path, 'rb') as f:
        files = {'file' : (exp_json_path, f, "application/json")}
        response = requests.post(url, files=files)


    try:
        result = response.json()
    except Exception as e:
        return {"status": 'error', "message": str(e)}
    
    return result

def get_args():
    parser = argparse.ArgumentParser(description='Train a feedforward neural network on MNIST using a JSON configuration file.')

    parser.add_argument('config', type=str, default='./exp.json' ,help='Path to the JSON Config file') 

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = get_args()
    json_path = args.config

    res = train_experiments(json_path)
    print(res)