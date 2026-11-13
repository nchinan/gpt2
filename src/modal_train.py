"""Run train.py on a Modal GPU and copy the trained model back to models/<prefix>-gpt2/<size>.

    uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --epochs 10
    uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --gpu A10G

Trained models are also kept in the Modal Volume "gpt2-models", and downloaded corpora in "gpt2-corpus".
"""
import os
import subprocess
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # local repo root
REMOTE = "/root/project"                                             # repo root inside the container

# Dependencies from pyproject.toml/uv.lock (cached between runs), then our source code on top
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(ROOT)
    .add_local_dir(os.path.join(ROOT, "src"), f"{REMOTE}/src", ignore=["**/__pycache__"])
)

models_volume = modal.Volume.from_name("gpt2-models", create_if_missing=True)
corpus_volume = modal.Volume.from_name("gpt2-corpus", create_if_missing=True)

app = modal.App("gpt2-train", image=image)


@app.function(
    gpu="L4",              # default; override with --gpu
    timeout=12 * 60 * 60,  # 12 hours
    volumes={f"{REMOTE}/models": models_volume, f"{REMOTE}/corpus": corpus_volume},
)
def train(model_prefix: str, size: str, corpus: str, epochs: int) -> str:
    """Run train.py in the container. Returns the saved model's folder relative to models/."""
    subprocess.run(
        [sys.executable, "src/train.py", "--model-prefix", model_prefix, "--model", "gpt2",
         "--size", size, "--corpus", corpus, "--epochs", str(epochs)],
        cwd=REMOTE,
        check=True,
    )
    models_volume.commit()
    corpus_volume.commit()

    sys.path.insert(0, f"{REMOTE}/src")
    from model_factory import MODELS_DIR, model_dir, resolve_size
    return os.path.relpath(model_dir("gpt2", resolve_size("gpt2", size), model_prefix), MODELS_DIR)


@app.local_entrypoint()
def main(model_prefix: str, size: str = "124M", corpus: str = "the-verdict", epochs: int = 10, gpu: str = "L4"):
    print(f"Training gpt2 {size} on {corpus} for {epochs} epochs on a Modal {gpu} GPU")
    saved = train.with_options(gpu=gpu).remote(model_prefix, size, corpus, epochs)

    # Copy the trained model from the volume to the local models/ folder
    local_dir = os.path.join(ROOT, "models", saved)
    os.makedirs(local_dir, exist_ok=True)
    for name in ["model.pth", "config.json"]:
        tmp = os.path.join(local_dir, name + ".part")
        with open(tmp, "wb") as f:
            for chunk in models_volume.read_file(f"{saved}/{name}"):
                f.write(chunk)
        os.replace(tmp, os.path.join(local_dir, name))
    print(f"Downloaded trained model to {local_dir}")
