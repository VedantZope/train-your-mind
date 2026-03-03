"""Generate binary keypoint patch masks (default: 3x3x3) from keypoint CSV files."""

from __future__ import annotations

from pathlib import Path

from mind.evotorch.data import build_pairs
from mind.utils.keypoint_masks import generate_keypoint_patch_mask


def _default_out_dir(root: Path, dataset: str, dataset_root: Path | None) -> Path:
    ds = dataset.lower()
    if ds == "lungct":
        return root / "data" / "LungCT_L2R" / "keypointMasksTr"
    if ds == "nlst":
        base = dataset_root or Path("/mnt/rohit_data2/NLST/NLST")
        return base / "keypointMasksTr"
    raise ValueError(f"Unsupported dataset: {dataset}")


def run(
    dataset: str,
    root: Path,
    dataset_root: Path | None,
    out_dir: Path | None,
    radius_vox: int,
) -> None:
    pairs = build_pairs(root=root, dataset=dataset, dataset_root=dataset_root)
    target_dir = out_dir or _default_out_dir(root, dataset, dataset_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    generated = 0
    for sample in pairs:
        for image_path, kp_csv in (
            (sample.fixed_path, sample.fixed_kp),
            (sample.moving_path, sample.moving_kp),
        ):
            key = image_path.name
            if key in seen:
                continue
            seen.add(key)
            if not kp_csv.exists():
                print(f"[WARN] missing keypoint csv for {image_path.name}: {kp_csv}")
                continue
            out_path = target_dir / image_path.name
            generate_keypoint_patch_mask(
                image_path=image_path,
                keypoint_csv=kp_csv,
                out_path=out_path,
                radius_vox=radius_vox,
            )
            generated += 1

    print(f"[DONE] Generated {generated} keypoint patch masks in {target_dir}")
    print(f"[INFO] radius_vox={radius_vox} => patch size={(2 * radius_vox + 1)}^3")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 3x3x3 keypoint patch masks")
    parser.add_argument(
        "--dataset",
        choices=["lungct", "nlst"],
        default="lungct",
        help="Dataset to process",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/vedant/MIND"),
        help="Project root (used for lungct defaults)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root override (used for nlst)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory override",
    )
    parser.add_argument(
        "--radius-vox",
        type=int,
        default=1,
        help="Patch radius in voxels (1 => 3x3x3)",
    )
    args = parser.parse_args()

    run(
        dataset=args.dataset,
        root=args.root,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        radius_vox=args.radius_vox,
    )
