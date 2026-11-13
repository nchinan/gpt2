## Introduction

A from-scratch GPT-2, extended step by step with LLM architecture improvements since GPT-2. The target is a small gpt-oss-style model: GPT-2 plus RMSNorm, RoPE, grouped-query attention, mixture-of-experts with SwiGLU, and alternating banded/dense attention. A ~100M version trains on WikiText-2 for about $1 on Modal.

## Usage

Pretrained weights are downloaded to `models/<model>/<size>` on first use.

### GPT-2

Sizes: `124M`, `355M`, `774M`, `1558M`.

```bash
# Generate with pretrained weights
uv run python src/generate.py --model gpt2 --size 124M --prompt "Every effort moves you"

# Train from random weights (prefix required), saved to models/<prefix>-gpt2/<size>
uv run python src/train.py --model-prefix mine --size 124M

# Train on a different corpus
uv run python src/train.py --model-prefix wiki --size 124M --corpus wikitext-2

# Set the number of passes over the corpus (default 10)
uv run python src/train.py --model-prefix wiki --size 124M --corpus wikitext-2 --epochs 1

# Generate with your trained model (uses random weights, with a warning, if not trained yet)
uv run python src/generate.py --model-prefix mine --model gpt2 --size 124M --prompt "Every effort moves you"
```

Training corpora (`--corpus`), downloaded to `corpus/<name>.txt` on first use:

| Corpus | Size | Notes |
|---|---|---|
| `the-verdict` (default) | ~5k tokens | Short story; trains in under a minute on Apple Silicon (MPS) |
| `wikitext-2` | ~2.4M tokens | Wikipedia articles, a GPT-2 paper benchmark; ~3 hours for 10 epochs on MPS |

Training uses CUDA if available, then Apple's MPS, then CPU.

#### Train on Modal (cloud GPU)

Takes the same options as `train.py`, plus `--gpu` (default `L4`). The trained model is copied back to `models/<prefix>-gpt2/<size>`, so `generate.py --model-prefix` works as normal.

```bash
# One-time login (opens the browser)
uv run modal token new

uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --epochs 10
uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --gpu A10G
```

The first run builds the image, which takes a few minutes. Trained models are also kept in the Modal Volume `gpt2-models`. See [MODAL.md](MODAL.md) for setup details and cost estimates.

### Llama 2

*Note* You need to request access on Hugging Face

Size: `7b`. Generation only. Requires a Hugging Face token and accepting Meta's license on the [model page](https://huggingface.co/meta-llama/Llama-2-7b).

```bash
export HF_ACCESS_TOKEN=hf_...
uv run python src/generate.py --model llama2 --size 7b --prompt "Every effort moves you"
```

### Download only

```bash
uv run python src/download.py --model gpt2 --size 124M
```

## Books

* [Mathematics for Machine Learning, 2020, Deisenroth, Cambridge University Press](https://mml-book.github.io/book/mml-book.pdf)

## Lecture Series

* [DeepMind x UCL | Deep Learning Lecture Series 2020](https://www.youtube.com/playlist?list=PLqYmG7hTraZCDxZ44o4p3N5Anz3lLRVZF)
* [DeepMind x UCL | Deep Learning Lecture Series 2021 (Reinforcement Learning)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)

## Papers

### Models

* Transformer [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
* GPT-1 [Improving Language Understanding by Generative Pre-Training (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* GPT-2 [Language Models are Unsupervised Multitask Learners (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* GPT-3 [Language Models are Few-Shot Learners (2020)](https://arxiv.org/abs/2005.14165)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)](https://arxiv.org/abs/2307.09288)
* gpt-oss [gpt-oss-120b & gpt-oss-20b Model Card (2025)](https://arxiv.org/abs/2508.10925)

### Optimisations

* [Layer Normalization (2016)](https://arxiv.org/abs/1607.06450)
* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)