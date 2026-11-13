# Training on Modal

`src/modal_train.py` runs `src/train.py` on a [Modal](https://modal.com) cloud GPU and copies the trained model back to `models/<prefix>-gpt2/<size>`, so `generate.py --model-prefix <prefix>` works as normal.

## Setup

Modal authenticates with a token pair, `MODAL_TOKEN_ID` (`ak-...`) and `MODAL_TOKEN_SECRET` (`as-...`). The easiest way to set this up is a one-time login, which opens the browser and saves the token to `~/.modal.toml`:

```bash
uv run modal token new
```

## Usage

Takes the same options as `train.py`, plus `--gpu` (default `L4`).

```bash
uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --epochs 10
uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --gpu A10G
```

- The first run builds the image (dependencies from `uv.lock`, then `src/`), which takes a few minutes. Later runs reuse it.
- Trained models are kept in the Modal Volume `gpt2-models`, and downloaded corpora in `gpt2-corpus`.
- Jobs time out after 12 hours.

## Cost

Prices from [modal.com/pricing](https://modal.com/pricing) (November 2026). The Starter plan includes $30/month of free credits.

| Resource | Price | Per hour |
|---|---|---|
| Nvidia T4 | $0.000164/sec | ~$0.59 |
| Nvidia L4 (default) | $0.000222/sec | ~$0.80 |
| Nvidia A10 (`--gpu A10G`) | $0.000306/sec | ~$1.10 |
| CPU | $0.0000131/core/sec | ~$0.05 per core |
| Memory | $0.00000222/GiB/sec | ~$0.008 per GiB |

### Estimate: WikiText-2, 10 epochs, A10G

```bash
uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --gpu A10G
```

**About $1 (roughly $0.60–$1.50).** This is an estimate; the A10G speed has not been measured.

- WikiText-2 is 4,201 batches per epoch, so 10 epochs is ~42,000 training steps.
- Apple Silicon (MPS) runs ~4.3 steps/sec. An A10G at batch size 2 without mixed precision is expected to run ~10–20 steps/sec, so 35–70 minutes.
- CPU and memory add under $0.10/hour. The first image build adds a few cents.

| Run time | Approx. cost |
|---|---|
| 35 min | ~$0.70 |
| 60 min | ~$1.20 |
| 75 min | ~$1.50 |

The worst case is capped by the 12-hour timeout at about $14.

### Measuring the real cost

Run one epoch first. The progress bar shows steps/sec, and the cost of 10 epochs is about 10× the one-epoch run:

```bash
uv run modal run src/modal_train.py --model-prefix wiki --corpus wikitext-2 --gpu A10G --epochs 1
```

At batch size 2 the model can't use the A10G's extra power, so the default L4 may be cheaper overall.
