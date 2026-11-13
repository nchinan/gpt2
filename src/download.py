import argparse

from model_factory import SIZES, download, resolve_size


def main():
    parser = argparse.ArgumentParser(description="Download pretrained model weights into models/<model>/<size>.")
    parser.add_argument("--model", required=True, choices=list(SIZES), help="Model family to download")
    parser.add_argument("--size", required=True,
                        help="; ".join(f"{m}: {', '.join(s)}" for m, s in SIZES.items()))
    args = parser.parse_args()

    try:
        size = resolve_size(args.model, args.size)
    except ValueError as e:
        parser.error(str(e))

    path = download(args.model, size)
    print(f"{args.model} {size} is in {path}")


if __name__ == "__main__":
    main()
