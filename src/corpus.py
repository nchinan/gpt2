"""Training corpora, downloaded once to corpus/<name>.txt."""
import os

from model_factory import download_file

# Repo-level corpus/ directory, independent of the current working directory
CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")

CORPORA = {
    # Short story by Edith Wharton (~5k tokens), used in "Build a Large Language Model (From Scratch)"
    "the-verdict": {
        "url": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
    },
    # Wikipedia "Good" and "Featured" articles (~2.4M tokens, training split).
    # One of the language-modelling benchmarks in the GPT-2 paper (WikiText-2, WikiText-103).
    "wikitext-2": {
        "url": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "parquet_column": "text",
    },
}


def load_corpus(name):
    """Return the text of corpus `name`, downloading it to corpus/<name>.txt on first use."""
    if name not in CORPORA:
        raise ValueError(f"Unknown corpus {name!r}; choose from: {', '.join(CORPORA)}")
    source = CORPORA[name]
    txt_path = os.path.join(CORPUS_DIR, f"{name}.txt")

    if not os.path.exists(txt_path):
        os.makedirs(CORPUS_DIR, exist_ok=True)
        if "parquet_column" in source:
            # Download the Parquet file, flatten its text column into one .txt, then remove the Parquet
            import pyarrow.parquet as pq

            parquet_path = os.path.join(CORPUS_DIR, f"{name}.parquet")
            download_file(source["url"], parquet_path)
            rows = pq.read_table(parquet_path, columns=[source["parquet_column"]]).column(0).to_pylist()
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("".join(rows))  # each WikiText row already ends with "\n"
            os.remove(parquet_path)
        else:
            download_file(source["url"], txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()
