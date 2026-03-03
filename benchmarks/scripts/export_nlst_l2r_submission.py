"""Export NLST Learn2Reg submission displacement fields using FireANTs.

Writes scipy-convention displacement fields with names:
  disp_<fixed_case4>_<moving_case4>.nii.gz
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import torch
from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image
from fireants.registration.greedy import GreedyRegistration

from mind.utils.io import mind_descriptor_image_convex


def _extract_case4(path_str: str) -> str:
    name = Path(path_str).name
    m = re.match(r"NLST_(\d{4})_\d{4}\.nii(\.gz)?$", name)
    if not m:
        raise ValueError(f"Could not parse NLST case id from: {name}")
    return m.group(1)


def _load_pairs(dataset_json: Path, split_key: str) -> list[dict[str, str]]:
    with dataset_json.open("r") as f:
        d = json.load(f)
    if split_key not in d:
        raise KeyError(f"{split_key} not found in {dataset_json}")
    pairs = d[split_key]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{split_key} is empty in {dataset_json}")
    return pairs


def _resolve_pair_paths(nlst_root: Path, pair: dict[str, str]) -> tuple[Path, Path, Path | None, Path | None]:
    fixed_rel = pair["fixed"]
    moving_rel = pair["moving"]
    fixed = (nlst_root / fixed_rel).resolve()
    moving = (nlst_root / moving_rel).resolve()

    fixed_mask = None
    moving_mask = None
    if "imagesTr" in fixed_rel:
        fixed_mask = (nlst_root / fixed_rel.replace("imagesTr", "masksTr")).resolve()
        moving_mask = (nlst_root / moving_rel.replace("imagesTr", "masksTr")).resolve()
    elif "imagesTs" in fixed_rel:
        fixed_mask = (nlst_root / fixed_rel.replace("imagesTs", "masksTs")).resolve()
        moving_mask = (nlst_root / moving_rel.replace("imagesTs", "masksTs")).resolve()

    return fixed, moving, fixed_mask, moving_mask


def run_export(
    nlst_root: Path,
    dataset_json: Path,
    split_key: str,
    out_dir: Path,
    masked: bool,
    use_mind: bool,
    cc_kernel_size: int,
    smooth_warp_sigma: float,
    smooth_grad_sigma: float,
    downsample_scale: int,
    make_zip: bool,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = _load_pairs(dataset_json, split_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, pair in enumerate(pairs, start=1):
        fixed_path, moving_path, fixed_mask_path, moving_mask_path = _resolve_pair_paths(nlst_root, pair)

        fixed_img = Image.load_file(str(fixed_path), device=device)
        moving_img = Image.load_file(str(moving_path), device=device)

        fixed_in = fixed_img
        moving_in = moving_img

        if use_mind:
            fixed_in = mind_descriptor_image_convex(fixed_in)
            moving_in = mind_descriptor_image_convex(moving_in)

        if masked:
            if fixed_mask_path is None or moving_mask_path is None:
                raise FileNotFoundError("Masked mode requested but mask paths could not be inferred")
            if not fixed_mask_path.exists() or not moving_mask_path.exists():
                raise FileNotFoundError(
                    f"Mask missing for pair: {fixed_mask_path} / {moving_mask_path}"
                )
            fixed_mask = Image.load_file(str(fixed_mask_path), device=device)
            moving_mask = Image.load_file(str(moving_mask_path), device=device)
            fixed_in = apply_mask_to_image(fixed_in, fixed_mask)
            moving_in = apply_mask_to_image(moving_in, moving_mask)

        reg = GreedyRegistration(
            scales=[4, 2, 1],
            iterations=[200, 100, 50],
            fixed_images=BatchedImages([fixed_in]),
            moving_images=BatchedImages([moving_in]),
            loss_type="masked_cc" if masked else "cc",
            cc_kernel_size=cc_kernel_size,
            smooth_warp_sigma=smooth_warp_sigma,
            smooth_grad_sigma=smooth_grad_sigma,
            max_tolerance_iters=1000,
            progress_bar=True,
        )
        reg.optimize()

        fixed_case4 = _extract_case4(pair["fixed"])
        moving_case4 = _extract_case4(pair["moving"])
        out_name = f"disp_{fixed_case4}_{moving_case4}.nii.gz"
        out_path = out_dir / out_name

        reg.save_as_scipy_transforms(
            str(out_path),
            downsample_scale=downsample_scale,
            dtype=torch.float32,
        )

        print(f"[{idx}/{len(pairs)}] saved {out_path.name}")

    if make_zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.glob("disp_*.nii.gz")):
                zf.write(p, arcname=f"{out_dir.name}/{p.name}")
        print(f"[DONE] zip written: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NLST Learn2Reg submission fields")
    parser.add_argument(
        "--nlst-root",
        type=Path,
        default=Path("/mnt/rohit_data2/NLST/NLST"),
        help="NLST dataset root",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="Path to NLST_dataset.json (default: <nlst-root>/NLST_dataset.json)",
    )
    parser.add_argument(
        "--split",
        choices=["registration_val", "registration_test"],
        default="registration_val",
        help="Which pair list from dataset json to export",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/vedant/MIND/benchmarks/nlst/l2r_submission_val"),
        help="Output directory for disp_XXXX_YYYY.nii.gz files",
    )
    parser.add_argument("--masked", action="store_true", help="Use masked_cc")
    parser.add_argument(
        "--no-mind",
        action="store_true",
        help="Disable MIND descriptor and register raw intensity",
    )
    parser.add_argument("--cc-kernel-size", type=int, default=19)
    parser.add_argument("--smooth-warp-sigma", type=float, default=0.943661231295491)
    parser.add_argument("--smooth-grad-sigma", type=float, default=3.977854595777292)
    parser.add_argument("--downsample-scale", type=int, default=1)
    parser.add_argument("--zip", action="store_true", help="Also create <out-dir>.zip")
    args = parser.parse_args()

    dataset_json = args.dataset_json or (args.nlst_root / "NLST_dataset.json")

    run_export(
        nlst_root=args.nlst_root,
        dataset_json=dataset_json,
        split_key=args.split,
        out_dir=args.out_dir,
        masked=args.masked,
        use_mind=not args.no_mind,
        cc_kernel_size=args.cc_kernel_size,
        smooth_warp_sigma=args.smooth_warp_sigma,
        smooth_grad_sigma=args.smooth_grad_sigma,
        downsample_scale=args.downsample_scale,
        make_zip=args.zip,
    )
