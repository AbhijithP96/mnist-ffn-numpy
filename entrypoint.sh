#!/bin/sh
set -e

# Start MLflow in the background
mlflow server --host 0.0.0.0 --port 8080 &

# Run uvicorn in the foreground (keeps container alive)
exec uv run uvicorn app:app --host 0.0.0.0 --port 5000
