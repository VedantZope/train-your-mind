"""Compute keypoint distances after GreedyRegistration for all pairs."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image
from fireants.io.keypoints import BatchedKeypoints, Keypoints, compute_keypoint_distance
from fireants.registration.greedy import GreedyRegistration

from mind.evotorch.data import build_pairs
from mind.utils.io import get_benchmark_paths, load_keypoints_csv


def _resolve_params_from_hparam_summary(
    dataset: str,
    cc_kernel_size: int | None,
    smooth_warp_sigma: float | None,
    smooth_grad_sigma: float | None,
) -> tuple[int, float, float]:
    # Keep legacy defaults as fallback if no summary exists.
    cc = 15
    sw = 0.5499451812055443
    sg = 3.6329020170516038

    summary_path = Path("/data/vedant/MIND") / "benchmarks" / dataset / "hparam_search" / "summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="") as f:
            row = next(csv.DictReader(f), None)
        if row is not None:
            cc = int(row["cc_width"])
            sw = float(row["sigma_warp"])
            sg = float(row["sigma_grad"])
            print(f"[PARAMS] Loaded defaults from {summary_path}")
        else:
            print(f"[PARAMS] Empty summary file at {summary_path}; using legacy defaults")
    else:
        print(f"[PARAMS] Summary not found at {summary_path}; using legacy defaults")

    if cc_kernel_size is not None:
        cc = cc_kernel_size
    if smooth_warp_sigma is not None:
        sw = smooth_warp_sigma
    if smooth_grad_sigma is not None:
        sg = smooth_grad_sigma
    print(f"[PARAMS] cc_kernel_size={cc} smooth_warp_sigma={sw} smooth_grad_sigma={sg}")
    return cc, sw, sg


def run_greedy_registration(
    masked: bool,
    dataset: str = "lungct",
    dataset_root: Path | None = None,
    cc_kernel_size: int | None = None,
    smooth_warp_sigma: float | None = None,
    smooth_grad_sigma: float | None = None,
) -> None:
    paths = get_benchmark_paths(Path("/data/vedant/MIND"))
    out_dir = paths.benchmark_root / dataset / (
        "greedy_registration_masked" if masked else "greedy_registration_nomask"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(Path("/data/vedant/MIND"), dataset=dataset, dataset_root=dataset_root)
    if not pairs:
        raise FileNotFoundError(f"No image pairs found for dataset={dataset}")
    cc_kernel_size, smooth_warp_sigma, smooth_grad_sigma = _resolve_params_from_hparam_summary(
        dataset=dataset,
        cc_kernel_size=cc_kernel_size,
        smooth_warp_sigma=smooth_warp_sigma,
        smooth_grad_sigma=smooth_grad_sigma,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_csv = out_dir / "keypoint_distances.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "fixed_image",
                "moving_image",
                "dist_initial_physical",
                "dist_final_physical",
            ]
        )

        for sample in pairs:
            case_id = sample.case_id
            fixed_path = sample.fixed_path
            moving_path = sample.moving_path

            fixed_img = Image.load_file(str(fixed_path), device=device)
            moving_img = Image.load_file(str(moving_path), device=device)

            if masked:
                fixed_mask = Image.load_file(str(sample.fixed_mask), device=device)
                moving_mask = Image.load_file(str(sample.moving_mask), device=device)
                fixed_img_in = apply_mask_to_image(fixed_img, fixed_mask)
                moving_img_in = apply_mask_to_image(moving_img, moving_mask)
            else:
                fixed_img_in = fixed_img
                moving_img_in = moving_img

            fixed_pts = load_keypoints_csv(sample.fixed_kp)
            moving_pts = load_keypoints_csv(sample.moving_kp)
            fixed_kp = Keypoints(fixed_pts, fixed_img, device=device, space="pixel")
            moving_kp = Keypoints(moving_pts, moving_img, device=device, space="pixel")

            fixed_batch = BatchedImages([fixed_img_in])
            moving_batch = BatchedImages([moving_img_in])
            fixed_kp_batch = BatchedKeypoints([fixed_kp])
            moving_kp_batch = BatchedKeypoints([moving_kp])

            initial_dist = compute_keypoint_distance(fixed_kp_batch, moving_kp_batch, space="physical", reduction="mean").item()

            reg = GreedyRegistration(
                scales=[4, 2, 1],
                iterations=[200, 100, 50],
                fixed_images=fixed_batch,
                moving_images=moving_batch,
                loss_type="masked_cc" if masked else "cc",
                cc_kernel_size=cc_kernel_size,
                smooth_warp_sigma=smooth_warp_sigma,
                smooth_grad_sigma=smooth_grad_sigma,
                progress_bar=True,
            )
            reg.optimize()

            moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)

            final_dist = compute_keypoint_distance(moved_kp_batch, moving_kp_batch, space="physical", reduction="mean").item()

            writer.writerow([case_id, fixed_path.name, moving_path.name, initial_dist, final_dist])
            print(f"[DONE] {case_id} initial={initial_dist:.4f} final={final_dist:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Greedy registration benchmark")
    parser.add_argument("--masked", action="store_true", help="Use masked CC loss")
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
    parser.add_argument(
        "--cc-kernel-size",
        type=int,
        default=None,
        help="Override cc kernel size (default from benchmarks/<dataset>/hparam_search/summary.csv)",
    )
    parser.add_argument(
        "--smooth-warp-sigma",
        type=float,
        default=None,
        help="Override smooth_warp_sigma (default from benchmarks/<dataset>/hparam_search/summary.csv)",
    )
    parser.add_argument(
        "--smooth-grad-sigma",
        type=float,
        default=None,
        help="Override smooth_grad_sigma (default from benchmarks/<dataset>/hparam_search/summary.csv)",
    )
    args = parser.parse_args()

    run_greedy_registration(
        masked=args.masked,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        cc_kernel_size=args.cc_kernel_size,
        smooth_warp_sigma=args.smooth_warp_sigma,
        smooth_grad_sigma=args.smooth_grad_sigma,
    )
