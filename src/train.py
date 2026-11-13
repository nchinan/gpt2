import argparse
import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from corpus import CORPORA, load_corpus
from gpt2.model import GPT2Model
from model_factory import GPT2_MODELS, gpt2_train_config, resolve_size, save_custom_model, validate_prefix
from model_stats import print_model_stats, print_total_params_bf16, print_total_params_f32


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model_simple(model, train_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for input_batch, target_batch in progress:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients

            progress.set_postfix(loss=f"{loss.item():.4f}")


def main(gpt_config, settings, corpus="the-verdict"):
    torch.manual_seed(123)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on: {device}")

    ##############################
    # Download data if necessary
    ##############################

    text_data = load_corpus(corpus)  # cached in corpus/<name>.txt
    ##############################
    # Initialize model
    ##############################

    model = GPT2Model(gpt_config)
    print_model_stats(model)
    print("Estimated training memory (params + grads + buffers):")
    print_total_params_f32(model)
    print_total_params_bf16(model)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    train_model_simple(model, train_loader, optimizer, device, num_epochs=settings["num_epochs"])

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Pretrain a model and save it to models/<prefix>-<model>/<size>.")
    parser.add_argument("--model-prefix", required=True,
                        help="Your name for this model; saved to models/<prefix>-<model>/<size>")
    parser.add_argument("--model", default="gpt2", choices=["gpt2"], help="Model architecture to train")
    parser.add_argument("--size", default="124M", help="gpt2: " + ", ".join(GPT2_MODELS))
    parser.add_argument("--corpus", default="the-verdict", choices=list(CORPORA),
                        help="Training text, downloaded to corpus/<name>.txt on first use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of passes over the training corpus")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    try:
        size = resolve_size(args.model, args.size)
        validate_prefix(args.model_prefix)
    except ValueError as e:
        parser.error(str(e))

    # Shared with generate.py so an untrained --model-prefix model has the same architecture
    GPT_CONFIG = gpt2_train_config(size)

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": args.epochs,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    model = main(GPT_CONFIG, OTHER_SETTINGS, corpus=args.corpus)

    path = save_custom_model(model, GPT_CONFIG, args.model, size, args.model_prefix)
    print(f"Saved model to {path}")
