"""Create pretrained models by family and size, downloading weights into models/<model>/<size> on first use."""
import json
import os
import re
import sys
from dataclasses import dataclass

import requests
import tiktoken
import torch
from tqdm import tqdm

from gpt2.model import GPT2Model
from llama2.model import Llama2Model, LLAMA2_CONFIG_7B
from llama2.tokenizer import LlamaTokenizer
from qwen3.model import Qwen3Model, QWEN_CONFIG_06_B
from qwen3.tokenizer import Qwen3Tokenizer

# Repo-level models/ directory, independent of the current working directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

##############################
# GPT-2
##############################

GPT2_BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

# size -> (weights file on Hugging Face, architecture config)
GPT2_MODELS = {
    "124M": ("gpt2-small-124M.pth", {"emb_dim": 768, "n_layers": 12, "n_heads": 12}),
    "355M": ("gpt2-medium-355M.pth", {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}),
    "774M": ("gpt2-large-774M.pth", {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}),
    "1558M": ("gpt2-xl-1558M.pth", {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}),
}
GPT2_URL = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

# Config used when training your own GPT-2 with train.py (and for an untrained model when none exists yet)
GPT2_TRAIN_BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}


def gpt2_train_config(size):
    return {**GPT2_TRAIN_BASE_CONFIG, **GPT2_MODELS[size][1]}

##############################
# Llama 2
##############################

# Only 7b: Llama2Model.load is written for the 7B layout, and 70b would also need grouped-query attention
LLAMA2_MODELS = {"7b": LLAMA2_CONFIG_7B}
LLAMA2_FILES = ["consolidated.00.pth", "params.json", "tokenizer.model"]

##############################
# Qwen3
##############################

# size -> (weights file, tokenizer file, architecture config). The .pth files are state dicts already
# in Qwen3Model's naming, so they load with load_state_dict (Qwen3Model.load is for HF safetensors).
QWEN3_MODELS = {
    "0.6b": ("qwen3-0.6B-base.pth", "tokenizer-base.json", QWEN_CONFIG_06_B),
    "0.6b-reasoning": ("qwen3-0.6B-reasoning.pth", "tokenizer-reasoning.json", QWEN_CONFIG_06_B),
}
QWEN3_URL = "https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/{file_name}"
QWEN3_BACKUP_URL = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B/{file_name}"

SIZES = {"gpt2": list(GPT2_MODELS), "llama2": list(LLAMA2_MODELS), "qwen3": list(QWEN3_MODELS)}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object  # has encode(text) -> list[int] and decode(list[int]) -> str
    config: dict


def resolve_size(model_name, size):
    """Return the canonical size name (e.g. '7B' -> '7b'), or raise ValueError."""
    if model_name not in SIZES:
        raise ValueError(f"Unknown model {model_name!r}; choose from: {', '.join(SIZES)}")
    for s in SIZES[model_name]:
        if s.lower() == size.lower():
            return s
    raise ValueError(f"Invalid size {size!r} for {model_name}; choose from: {', '.join(SIZES[model_name])}")


def validate_prefix(prefix):
    """Raise ValueError unless prefix is a safe single folder-name component (letters, digits, _ . -)."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix) or prefix in (".", ".."):
        raise ValueError(f"Invalid model prefix {prefix!r}; use only letters, digits, '_', '.' and '-'")
    return prefix


def model_dir(model_name, size, prefix=None):
    """models/<model>/<size> for pretrained weights, models/<prefix>-<model>/<size> for your own trained models."""
    folder = f"{validate_prefix(prefix)}-{model_name}" if prefix else model_name
    return os.path.join(MODELS_DIR, folder, size)


def download_file(url, dest):
    # Write to a temporary file first so an interrupted download is never mistaken for a complete one
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(desc=f"Fetching {os.path.basename(dest)}", total=total,
                                        unit="B", unit_scale=True, ncols=100) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    os.replace(tmp, dest)


def download_with_backup(url, backup_url, dest):
    try:
        download_file(url, dest)
    except requests.RequestException as e:
        print(f"Download from {url} failed ({e}); trying {backup_url}", file=sys.stderr)
        download_file(backup_url, dest)


def download_gpt2(size):
    directory = model_dir("gpt2", size)
    file_name = GPT2_MODELS[size][0]
    path = os.path.join(directory, file_name)
    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        download_file(GPT2_URL.format(file_name=file_name), path)
    return directory


def download_llama2(size):
    directory = model_dir("llama2", size)
    if all(os.path.exists(os.path.join(directory, f)) for f in LLAMA2_FILES):
        return directory

    from huggingface_hub import hf_hub_download, login

    access_token = os.environ.get("HF_ACCESS_TOKEN")
    if not access_token:
        sys.exit("HF_ACCESS_TOKEN is not set. Create a token at "
                 "https://huggingface.co/settings/tokens and export it first.")
    login(token=access_token)

    for filename in LLAMA2_FILES:
        hf_hub_download(repo_id=f"meta-llama/Llama-2-{size}", filename=filename, local_dir=directory)
    return directory


def download_qwen3(size, tokenizer_only=False):
    directory = model_dir("qwen3", size)
    model_file, tokenizer_file, _ = QWEN3_MODELS[size]
    targets = [tokenizer_file] if tokenizer_only else [model_file, tokenizer_file]

    os.makedirs(directory, exist_ok=True)
    for file_name in targets:
        path = os.path.join(directory, file_name)
        if not os.path.exists(path):
            download_with_backup(QWEN3_URL.format(file_name=file_name),
                                 QWEN3_BACKUP_URL.format(file_name=file_name), path)
    return directory


def _create_gpt2(size, device):
    file_name, arch = GPT2_MODELS[size]
    directory = download_gpt2(size)

    config = {**GPT2_BASE_CONFIG, **arch}
    model = GPT2Model(config)

    state_dict = torch.load(os.path.join(directory, file_name), map_location="cpu", weights_only=True)
    # The pretrained checkpoints name the attention layer "att"; our TransformerBlock calls it "attention"
    state_dict = {k.replace(".att.", ".attention."): v for k, v in state_dict.items()}
    if any(k.endswith(".attention.qkv.weight") for k in model.state_dict()):
        state_dict = _to_fused_qkv(state_dict)
    model.load_state_dict(state_dict)

    return LoadedModel(model.to(device).eval(), tiktoken.get_encoding("gpt2"), config)


def _to_fused_qkv(state_dict):
    """Convert checkpoint attention weights from MultiHeadAttention's layout to PyTorchMultiHeadAttention's.

    The checkpoints have separate W_query/W_key/W_value layers, out_proj and a causal-mask buffer.
    PyTorchMultiHeadAttention has one qkv layer (query, key, value stacked in that order), proj, and no mask.
    """
    converted = {}
    for key, value in state_dict.items():
        if key.endswith(".attention.mask") or ".attention.W_key." in key or ".attention.W_value." in key:
            continue  # mask isn't needed; key/value are merged in with the query below
        if ".attention.W_query." in key:
            prefix, param = key.split(".attention.W_query.")
            converted[f"{prefix}.attention.qkv.{param}"] = torch.cat([
                state_dict[f"{prefix}.attention.W_query.{param}"],
                state_dict[f"{prefix}.attention.W_key.{param}"],
                state_dict[f"{prefix}.attention.W_value.{param}"],
            ], dim=0)
        else:
            converted[key.replace(".attention.out_proj.", ".attention.proj.")] = value
    return converted


def _create_llama2(size, device):
    directory = download_llama2(size)

    config = LLAMA2_MODELS[size]
    model = Llama2Model(config)

    with open(os.path.join(directory, "params.json")) as f:
        param_config = json.load(f)
    # mmap avoids holding a second full copy of the ~13 GB checkpoint in memory
    params = torch.load(os.path.join(directory, "consolidated.00.pth"),
                        map_location="cpu", weights_only=True, mmap=True)
    Llama2Model.load(model, param_config, params)
    del params

    tokenizer = LlamaTokenizer(os.path.join(directory, "tokenizer.model"))
    return LoadedModel(model.to(device).eval(), tokenizer, config)


def _create_qwen3(size, device):
    directory = download_qwen3(size)
    model_file, tokenizer_file, config = QWEN3_MODELS[size]

    model = Qwen3Model(config)
    state_dict = torch.load(os.path.join(directory, model_file), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    reasoning = size.endswith("-reasoning")
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=os.path.join(directory, tokenizer_file),
        apply_chat_template=reasoning,
        add_generation_prompt=reasoning,
        add_thinking=reasoning,
    )
    return LoadedModel(model.to(device).eval(), tokenizer, config)


##############################
# Custom (prefixed) models trained with train.py
##############################

# Saved as models/<prefix>-<model>/<size>/{model.pth, config.json}. The config is stored because
# train.py uses a different architecture config (e.g. 256 context, no qkv bias) from the pretrained weights.
CUSTOM_WEIGHTS = "model.pth"
CUSTOM_CONFIG = "config.json"

# Model families that can be trained with train.py and loaded back with a prefix
_CUSTOM_CLASSES = {"gpt2": GPT2Model}
_TRAIN_CONFIGS = {"gpt2": gpt2_train_config}
_TOKENIZERS = {"gpt2": lambda: tiktoken.get_encoding("gpt2")}


def save_custom_model(model, config, model_name, size, prefix):
    """Save a trained model and its config to models/<prefix>-<model>/<size>. Returns the directory."""
    directory = model_dir(model_name, size, prefix)
    os.makedirs(directory, exist_ok=True)
    # torch.compile wraps the model as `_orig_mod`; save the original so keys have no "_orig_mod." prefix
    model = getattr(model, "_orig_mod", model)
    torch.save(model.state_dict(), os.path.join(directory, CUSTOM_WEIGHTS))
    with open(os.path.join(directory, CUSTOM_CONFIG), "w") as f:
        json.dump(config, f, indent=2)
    return directory


def _create_custom(model_name, size, prefix, device):
    if model_name not in _CUSTOM_CLASSES:
        raise ValueError(f"Custom {model_name} models are not supported; only: {', '.join(_CUSTOM_CLASSES)}")

    directory = model_dir(model_name, size, prefix)
    weights_path = os.path.join(directory, CUSTOM_WEIGHTS)
    config_path = os.path.join(directory, CUSTOM_CONFIG)
    # Never download into a prefixed folder: it only ever holds models you trained yourself.
    # If nothing has been trained yet, fall back to an untrained model so you can see the "before" output.
    if not (os.path.exists(weights_path) and os.path.exists(config_path)):
        print(f"Warning: no trained model in {directory}; using random weights. "
              f"Train one with: train.py --model-prefix {prefix} --model {model_name} --size {size}",
              file=sys.stderr)
        config = _TRAIN_CONFIGS[model_name](size)
        model = _CUSTOM_CLASSES[model_name](config)
    else:
        with open(config_path) as f:
            config = json.load(f)
        model = _CUSTOM_CLASSES[model_name](config)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Models saved from a torch.compile'd model before save_custom_model unwrapped it have this prefix
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    return LoadedModel(model.to(device).eval(), _TOKENIZERS[model_name](), config)


_DOWNLOADERS = {"gpt2": download_gpt2, "llama2": download_llama2, "qwen3": download_qwen3}
_CREATORS = {"gpt2": _create_gpt2, "llama2": _create_llama2, "qwen3": _create_qwen3}


def download(model_name, size):
    """Ensure the weights for model_name/size are in models/, downloading if missing. Returns the directory."""
    size = resolve_size(model_name, size)
    return _DOWNLOADERS[model_name](size)


def create_model(model_name, size, device="cpu", prefix=None):
    """Build a model and return it with its tokenizer and config.

    Without a prefix: the pretrained model from models/<model>/<size>, downloaded if needed.
    With a prefix:    your trained model from models/<prefix>-<model>/<size>, never downloaded.
    """
    size = resolve_size(model_name, size)
    if prefix:
        return _create_custom(model_name, size, prefix, torch.device(device))
    return _CREATORS[model_name](size, torch.device(device))
