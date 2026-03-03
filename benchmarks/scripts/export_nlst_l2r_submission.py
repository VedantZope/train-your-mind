"""Export NLST Learn2Reg submission displacement fields using FireANTs.

Writes scipy-convention displacement fields with names:
  disp_<fixed_case4>_<moving_case4>.nii.gz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image
from fireants.io.keypoints import BatchedKeypoints, Keypoints, compute_keypoint_distance
from fireants.registration.greedy import GreedyRegistration

from mind.evotorch.model import build_model
from mind.utils.io import image_from_tensor_like, load_keypoints_csv, mind_descriptor_image_convex


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


def _resolve_registration_val_paths(
    nlst_root: Path, pair: dict[str, str], image_dir_name: str
) -> tuple[Path, Path, Path, Path, Path | None, Path | None]:
    fixed_name = Path(pair["fixed"]).name
    moving_name = Path(pair["moving"]).name

    fixed_img = (nlst_root / image_dir_name / fixed_name).resolve()
    moving_img = (nlst_root / image_dir_name / moving_name).resolve()
    fixed_kp = (nlst_root / "keypointsTr" / f"{Path(fixed_name).stem.replace('.nii', '')}.csv").resolve()
    moving_kp = (nlst_root / "keypointsTr" / f"{Path(moving_name).stem.replace('.nii', '')}.csv").resolve()
    fixed_mask = (nlst_root / "masksTr" / fixed_name).resolve()
    moving_mask = (nlst_root / "masksTr" / moving_name).resolve()
    return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask


def _load_model_from_weights(weights_path: Path) -> torch.nn.Module:
    run_dir = weights_path.parent
    if run_dir.name == "checkpoints":
        run_dir = run_dir.parent
    cfg_path = run_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing model_config.json next to weights: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = json.load(f)
    model_name = cfg.get("model_name", "basic")
    model_params = cfg.get("model_params", {})
    model, _model_cfgs, _ = build_model(model_name, model_params)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _feature_image_model(model: torch.nn.Module, src_img: Image, device: str) -> Image:
    model = model.to(device)
    with torch.no_grad():
        tensor = src_img.array.float()
        if tensor.shape[1] > 1:
            tensor = tensor[:, :1]
        feat = model(tensor)
    return image_from_tensor_like(src_img, feat)


def run_export(
    nlst_root: Path,
    dataset_json: Path,
    out_dir: Path,
    masked: bool,
    image_dir_name: str,
    feature_mode: str,
    model_weights: Path | None,
    cc_kernel_size: int,
    smooth_warp_sigma: float,
    smooth_grad_sigma: float,
    downsample_scale: int,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = _load_pairs(dataset_json, "registration_val")
    out_dir.mkdir(parents=True, exist_ok=True)
    model: torch.nn.Module | None = None
    if feature_mode == "learned":
        if model_weights is None:
            raise ValueError("--model-weights is required when --feature-mode learned")
        model = _load_model_from_weights(model_weights).to(device).eval()
    tre_rows: list[tuple[str, str, str, float, float]] = []

    for idx, pair in enumerate(pairs, start=1):
        (
            fixed_path,
            moving_path,
            fixed_kp_path,
            moving_kp_path,
            fixed_mask_path,
            moving_mask_path,
        ) = _resolve_registration_val_paths(nlst_root, pair, image_dir_name=image_dir_name)

        if not fixed_path.exists() or not moving_path.exists():
            raise FileNotFoundError(
                f"Image file missing for pair {pair}: {fixed_path} / {moving_path}"
            )
        if not fixed_kp_path.exists() or not moving_kp_path.exists():
            print(
                f"[WARN] Missing keypoints for TRE print: {fixed_kp_path} / {moving_kp_path}"
            )
            fixed_kp_path = None
            moving_kp_path = None

        fixed_img = Image.load_file(str(fixed_path), device=device)
        moving_img = Image.load_file(str(moving_path), device=device)

        fixed_in = fixed_img
        moving_in = moving_img

        if feature_mode == "mind":
            fixed_in = mind_descriptor_image_convex(fixed_in)
            moving_in = mind_descriptor_image_convex(moving_in)
        elif feature_mode == "learned":
            if model is None:
                raise RuntimeError("learned mode requested but model is not initialized")
            fixed_in = _feature_image_model(model, fixed_in, device)
            moving_in = _feature_image_model(model, moving_in, device)

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

        tre_suffix = " | TRE unavailable (no keypoints for this split/pair)"
        if fixed_kp_path is not None and moving_kp_path is not None:
            fixed_pts = load_keypoints_csv(fixed_kp_path).astype("float32")
            moving_pts = load_keypoints_csv(moving_kp_path).astype("float32")
            fixed_kp = Keypoints(fixed_pts, fixed_img, device=device, space="pixel")
            moving_kp = Keypoints(moving_pts, moving_img, device=device, space="pixel")
            fixed_kp_batch = BatchedKeypoints([fixed_kp])
            moving_kp_batch = BatchedKeypoints([moving_kp])
            initial_dist = compute_keypoint_distance(
                fixed_kp_batch, moving_kp_batch, space="physical", reduction="mean"
            ).item()
            moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
            final_dist = compute_keypoint_distance(
                moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
            ).item()
            tre_suffix = f" | TRE init={initial_dist:.4f} final={final_dist:.4f}"
            tre_rows.append(
                (f"NLST_{fixed_case4}", fixed_path.name, moving_path.name, initial_dist, final_dist)
            )

        out_name = f"disp_{fixed_case4}_{moving_case4}.nii.gz"
        out_path = out_dir / out_name

        reg.save_as_scipy_transforms(
            str(out_path),
            downsample_scale=downsample_scale,
            dtype=torch.float32,
        )

        print(f"[{idx}/{len(pairs)}] saved {out_path.name}{tre_suffix}")

    if tre_rows:
        mean_initial = sum(r[3] for r in tre_rows) / len(tre_rows)
        mean_final = sum(r[4] for r in tre_rows) / len(tre_rows)
        print("\n=== TRE Summary (mm) ===")
        print(f"Pairs with keypoints: {len(tre_rows)} / {len(pairs)}")
        print(f"Mean initial TRE: {mean_initial:.6f}")
        print(f"Mean final TRE:   {mean_final:.6f}")
        print("\n=== TRE Table ===")
        print("case,fixed_image,moving_image,dist_initial_physical,dist_final_physical")
        for case_id, fixed_name, moving_name, initial_dist, final_dist in tre_rows:
            print(f"{case_id},{fixed_name},{moving_name},{initial_dist:.6f},{final_dist:.6f}")
    else:
        print("\n=== TRE Summary (mm) ===")
        print("No keypoint pairs found; TRE table is unavailable.")

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
        "--image-dir",
        type=str,
        default="imagesProcessedTr",
        help="Image folder to read registration_val pairs from (default: imagesProcessedTr)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/vedant/MIND/benchmarks/nlst/l2r_submission_val"),
        help="Output directory for disp_XXXX_YYYY.nii.gz files",
    )
    parser.add_argument("--masked", action="store_true", help="Use masked_cc")
    parser.add_argument(
        "--feature-mode",
        choices=["mind", "intensity", "learned"],
        default="mind",
        help="Feature mode for registration input",
    )
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=None,
        help="Path to learned model checkpoint (.pt/.pth); required for --feature-mode learned",
    )
    parser.add_argument(
        "--no-mind",
        action="store_true",
        help="Backward-compat: equivalent to --feature-mode intensity",
    )
    parser.add_argument("--cc-kernel-size", type=int, default=19)
    parser.add_argument("--smooth-warp-sigma", type=float, default=0.943661231295491)
    parser.add_argument("--smooth-grad-sigma", type=float, default=3.977854595777292)
    parser.add_argument("--downsample-scale", type=int, default=1)
    args = parser.parse_args()

    dataset_json = args.dataset_json or (args.nlst_root / "NLST_dataset.json")
    feature_mode = args.feature_mode
    if args.no_mind:
        feature_mode = "intensity"

    run_export(
        nlst_root=args.nlst_root,
        dataset_json=dataset_json,
        out_dir=args.out_dir,
        masked=args.masked,
        image_dir_name=args.image_dir,
        feature_mode=feature_mode,
        model_weights=args.model_weights,
        cc_kernel_size=args.cc_kernel_size,
        smooth_warp_sigma=args.smooth_warp_sigma,
        smooth_grad_sigma=args.smooth_grad_sigma,
        downsample_scale=args.downsample_scale,
    )
