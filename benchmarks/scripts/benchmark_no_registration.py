"""Compute baseline keypoint distances without registration."""

from __future__ import annotations

import csv
from pathlib import Path

from fireants.io.image import Image
from mind.evotorch.data import build_pairs

from mind.utils.io import (
    compute_keypoint_loss_from_csv,
    get_benchmark_paths,
)


def main(dataset: str = "lungct", dataset_root: Path | None = None) -> None:
    paths = get_benchmark_paths(Path("/data/vedant/MIND"))
    out_dir = paths.benchmark_root / dataset / "no_registration"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(Path("/data/vedant/MIND"), dataset=dataset, dataset_root=dataset_root)
    if not pairs:
        raise FileNotFoundError(f"No image pairs found for dataset={dataset}")

    out_csv = out_dir / "keypoint_distances.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case", "fixed_image", "moving_image", "dist_physical", "dist_pixel"]
        )

        for sample in pairs:
            case_id = sample.case_id
            fixed_path = sample.fixed_path
            moving_path = sample.moving_path

            fixed_img = Image.load_file(str(fixed_path), device="cpu")
            moving_img = Image.load_file(str(moving_path), device="cpu")

            dist_phys = compute_keypoint_loss_from_csv(
                fixed_img, moving_img, sample.fixed_kp, sample.moving_kp, space="physical"
            )
            dist_pix = compute_keypoint_loss_from_csv(
                fixed_img, moving_img, sample.fixed_kp, sample.moving_kp, space="pixel"
            )

            writer.writerow(
                [case_id, fixed_path.name, moving_path.name, dist_phys, dist_pix]
            )
            print(f"[DONE] {case_id} phys={dist_phys:.4f} pix={dist_pix:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="No-registration benchmark")
    parser.add_argument(
        "--dataset",
        choices=["lungct", "nlst"],
        default="lungct",
        help="Dataset loader to use",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root override (for NLST, default is /mnt/rohit_data2/NLST/NLST)",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, dataset_root=args.dataset_root)
