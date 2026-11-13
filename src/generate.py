import argparse
import time

import torch

from model_factory import SIZES, create_model, resolve_size, validate_prefix
from model_stats import print_model_stats

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    if eos_id is not None and idx.shape[0] != 1:
        raise ValueError("EOS stopping currently supports batch size 1 only")

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Generate text with a pretrained model.")
    parser.add_argument("--model", default="gpt2", choices=list(SIZES), help="Model family")
    parser.add_argument("--size", default="124M",
                        help="; ".join(f"{m}: {', '.join(s)}" for m, s in SIZES.items()))
    parser.add_argument("--model-prefix", default=None,
                        help="Load your trained model from models/<prefix>-<model>/<size> "
                             "instead of the pretrained models/<model>/<size>")
    parser.add_argument(
        "--prompt",
        default="Every effort moves you",
        help="Prompt text used to seed the generation."
    )

    args = parser.parse_args()

    try:
        size = resolve_size(args.model, args.size)
        if args.model_prefix:
            validate_prefix(args.model_prefix)
    except ValueError as e:
        parser.error(str(e))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    start = time.perf_counter()
    try:
        loaded = create_model(args.model, size, device=device, prefix=args.model_prefix)
    except (FileNotFoundError, ValueError) as e:
        parser.exit(1, f"error: {e}\n")
    load_seconds = time.perf_counter() - start  # includes the download on first use

    print(f"Model: {args.model_prefix + '-' if args.model_prefix else ''}{args.model} {size}")
    print(f"Load time: {load_seconds:.2f}s")
    print_model_stats(loaded.model)
    tokenizer = loaded.tokenizer
    torch.manual_seed(123)

    prompt_ids = text_to_token_ids(args.prompt, tokenizer).to(device)

    start = time.perf_counter()
    token_ids = generate(
        model=loaded.model,
        idx=prompt_ids,
        max_new_tokens=25,
        context_size=loaded.config["context_length"],
        top_k=50,
        temperature=1.0
    )
    # GPU work runs asynchronously; wait for it to finish before stopping the timer
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    generate_seconds = time.perf_counter() - start

    new_tokens = token_ids.shape[1] - prompt_ids.shape[1]
    print(f"Generation time: {generate_seconds:.2f}s "
          f"({new_tokens} tokens, {new_tokens / generate_seconds:.1f} tokens/s)")

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))