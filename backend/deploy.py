"""
Modal deployment for the Saddle backend.

Deploys the FastAPI app as a web endpoint on Modal's infrastructure.
The C shared library is built inside the container image.

Deploy:   modal deploy deploy.py
Serve:    modal serve deploy.py   (dev mode with hot reload)
"""

import modal

app = modal.App("saddle")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "make")
    .pip_install("fastapi>=0.115", "uvicorn>=0.34", "jax[cpu]>=0.5", "numpy>=1.26")
    .add_local_dir("csrc", remote_path="/app/csrc", copy=True)
    .add_local_dir("saddle", remote_path="/app/saddle", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    .run_commands(
        "cd /app/csrc && make",
        "cd /app && pip install -e .",
    )
)


@app.function(image=image, min_containers=1)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from saddle.api import app as fastapi_app
    return fastapi_app
