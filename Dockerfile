FROM python:3.12.3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
COPY . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

RUN uv pip install --system mlflow
RUN uv pip install PyYAML lark python-multipart

EXPOSE 5000 8080

# Copy your entrypoint script
RUN chmod +x /app/entrypoint.sh

# Use the script as entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

