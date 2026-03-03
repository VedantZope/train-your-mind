from mind.utils.io import preprocess_dataset_split


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preprocess LungCT_L2R dataset")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["Tr", "Ts", "both"],
        default="Tr",
        help="Which split to preprocess",
    )
    args = parser.parse_args()

    if args.split == "both":
        preprocess_dataset_split(args.root, args.out, split="Tr")
        preprocess_dataset_split(args.root, args.out, split="Ts")
    else:
        preprocess_dataset_split(args.root, args.out, split=args.split)


if __name__ == "__main__":
    main()
