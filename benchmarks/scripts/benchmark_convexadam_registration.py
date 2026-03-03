"""Compute keypoint distances after ConvexAdam registration for all pairs."""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from fireants.io.image import Image
from fireants.io.keypoints import BatchedKeypoints, Keypoints, compute_keypoint_distance
from scipy.ndimage import map_coordinates

from convexAdam.convex_adam_MIND import convex_adam_pt
from mind.evotorch.data import build_pairs
from mind.utils.io import get_benchmark_paths, load_keypoints_csv


def _sample_displacements_at_points(
    disp_field: np.ndarray,
    points_ijk: np.ndarray,
) -> np.ndarray:
    """Trilinear sample dense displacement field at point locations."""
    sampled = [
        map_coordinates(
            disp_field[..., c],
            [points_ijk[:, 0], points_ijk[:, 1], points_ijk[:, 2]],
            order=1,
            mode="nearest",
        )
        for c in range(3)
    ]
    return np.stack(sampled, axis=1)


def run_convexadam_registration(
    masked: bool,
    dataset: str,
    dataset_root: Path | None,
    mind_r: int,
    mind_d: int,
    lambda_weight: float,
    grid_sp: int,
    disp_hw: int,
    selected_niter: int,
    selected_smooth: int,
    grid_sp_adam: int,
    ic: bool,
) -> None:
    paths = get_benchmark_paths(Path("/data/vedant/MIND"))
    out_dir = paths.benchmark_root / dataset / (
        "convexadam_registration_masked" if masked else "convexadam_registration_nomask"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(out_dir)

    pairs = build_pairs(Path("/data/vedant/MIND"), dataset=dataset, dataset_root=dataset_root)
    if not pairs:
        raise FileNotFoundError(f"No image pairs found for dataset={dataset}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ca_dtype = torch.float32

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

            fixed_pts = load_keypoints_csv(sample.fixed_kp).astype(np.float32)
            moving_pts = load_keypoints_csv(sample.moving_kp).astype(np.float32)

            fixed_kp = Keypoints(fixed_pts, fixed_img, device=device, space="pixel")
            moving_kp = Keypoints(moving_pts, moving_img, device=device, space="pixel")
            fixed_kp_batch = BatchedKeypoints([fixed_kp])
            moving_kp_batch = BatchedKeypoints([moving_kp])

            initial_dist = compute_keypoint_distance(
                fixed_kp_batch,
                moving_kp_batch,
                space="physical",
                reduction="mean",
            ).item()

            fixed_np = np.ascontiguousarray(nib.load(str(fixed_path)).get_fdata(dtype=np.float32))
            moving_np = np.ascontiguousarray(nib.load(str(moving_path)).get_fdata(dtype=np.float32))

            disp_field = convex_adam_pt(
                img_fixed=fixed_np,
                img_moving=moving_np,
                mind_r=mind_r,
                mind_d=mind_d,
                lambda_weight=lambda_weight,
                grid_sp=grid_sp,
                disp_hw=disp_hw,
                selected_niter=selected_niter,
                selected_smooth=selected_smooth,
                grid_sp_adam=grid_sp_adam,
                ic=ic,
                use_mask=masked,
                path_fixed_mask=str(sample.fixed_mask) if masked else None,
                path_moving_mask=str(sample.moving_mask) if masked else None,
                dtype=ca_dtype,
                device=torch.device(device),
                verbose=False,
            )
            if not np.isfinite(disp_field).all():
                print(f"[WARN] {case_id} ConvexAdam produced non-finite displacement; setting final=nan")
                writer.writerow([case_id, fixed_path.name, moving_path.name, initial_dist, float("nan")])
                continue

            disp_at_fixed = _sample_displacements_at_points(disp_field, fixed_pts)
            fixed_pts_warped = fixed_pts + disp_at_fixed

            moved_fixed_kp = Keypoints(
                fixed_pts_warped.astype(np.float32),
                moving_img,
                device=device,
                space="pixel",
            )
            moved_fixed_kp_batch = BatchedKeypoints([moved_fixed_kp])

            final_dist = compute_keypoint_distance(
                moved_fixed_kp_batch,
                moving_kp_batch,
                space="physical",
                reduction="mean",
            ).item()

            writer.writerow([case_id, fixed_path.name, moving_path.name, initial_dist, final_dist])
            print(f"[DONE] {case_id} initial={initial_dist:.4f} final={final_dist:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConvexAdam registration benchmark")
    parser.add_argument("--masked", action="store_true", help="Use masks for ConvexAdam MIND")
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
    parser.add_argument("--mind-r", type=int, default=1)
    parser.add_argument("--mind-d", type=int, default=2)
    parser.add_argument("--lambda-weight", type=float, default=1.25)
    parser.add_argument("--grid-sp", type=int, default=6)
    parser.add_argument("--disp-hw", type=int, default=4)
    parser.add_argument("--selected-niter", type=int, default=80)
    parser.add_argument("--selected-smooth", type=int, default=0)
    parser.add_argument("--grid-sp-adam", type=int, default=2)
    parser.add_argument("--no-ic", action="store_true", help="Disable inverse consistency")
    args = parser.parse_args()

    run_convexadam_registration(
        masked=args.masked,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        mind_r=args.mind_r,
        mind_d=args.mind_d,
        lambda_weight=args.lambda_weight,
        grid_sp=args.grid_sp,
        disp_hw=args.disp_hw,
        selected_niter=args.selected_niter,
        selected_smooth=args.selected_smooth,
        grid_sp_adam=args.grid_sp_adam,
        ic=not args.no_ic,
    )
