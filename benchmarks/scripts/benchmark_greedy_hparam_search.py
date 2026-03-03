"""Bayesian hyperparameter search for GreedyRegistration (masked CC)."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import torch
from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image
from fireants.io.keypoints import BatchedKeypoints, Keypoints, compute_keypoint_distance
from fireants.registration.greedy import GreedyRegistration
import ray
from ray import tune
from ray.air import session
from ray.tune import RunConfig
import os
from ray.tune.search.hyperopt import HyperOptSearch

from mind.evotorch.data import build_pairs
from mind.utils.io import get_benchmark_paths, load_keypoints_csv


def evaluate_config(config: dict) -> None:
    dataset = config.get("dataset", "lungct")
    dataset_root = config.get("dataset_root")
    root = Path("/data/vedant/MIND")
    pairs = build_pairs(
        root,
        dataset=dataset,
        dataset_root=Path(dataset_root) if dataset_root else None,
    )
    if not pairs:
        raise FileNotFoundError(f"No image pairs found for dataset={dataset}")

    gpu_ids = ray.get_gpu_ids()
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(i)) for i in gpu_ids)
        device = "cuda"
    else:
        device = "cpu"

    final_dists = []

    for sample in pairs:
        fixed_path = sample.fixed_path
        moving_path = sample.moving_path

        fixed_img = Image.load_file(str(fixed_path), device=device)
        moving_img = Image.load_file(str(moving_path), device=device)

        fixed_mask = Image.load_file(str(sample.fixed_mask), device=device)
        moving_mask = Image.load_file(str(sample.moving_mask), device=device)

        fixed_img_masked = apply_mask_to_image(fixed_img, fixed_mask)
        moving_img_masked = apply_mask_to_image(moving_img, moving_mask)

        fixed_pts = load_keypoints_csv(sample.fixed_kp)
        moving_pts = load_keypoints_csv(sample.moving_kp)
        fixed_kp = Keypoints(fixed_pts, fixed_img, device=device, space="pixel")
        moving_kp = Keypoints(moving_pts, moving_img, device=device, space="pixel")

        fixed_batch = BatchedImages([fixed_img_masked])
        moving_batch = BatchedImages([moving_img_masked])
        fixed_kp_batch = BatchedKeypoints([fixed_kp])
        moving_kp_batch = BatchedKeypoints([moving_kp])

        reg = GreedyRegistration(
            scales=[4, 2, 1],
            iterations=[200, 100, 50],
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            loss_type="masked_cc",
            cc_kernel_size=int(config["cc_width"]),
            smooth_warp_sigma=float(config["sigma_warp"]),
            smooth_grad_sigma=float(config["sigma_grad"]),
            progress_bar=False,
        )
        reg.optimize()

        moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
        final_dist = compute_keypoint_distance(
            moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
        ).item()
        final_dists.append(final_dist)

    mean_final = mean(final_dists)
    session.report({"mean_final_dist": mean_final})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Greedy hparam search benchmark")
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
        "--num-samples",
        type=int,
        default=20,
        help="Number of sampled hparam trials",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=6,
        help="Maximum concurrent Ray Tune trials",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for HyperOpt sampler",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Force-evaluate known baseline params (cc=15, sigma_warp=0.5499451812, sigma_grad=3.6329020171)",
    )
    args = parser.parse_args()

    paths = get_benchmark_paths(Path("/data/vedant/MIND"))
    output_dir = paths.benchmark_root / args.dataset / "hparam_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = {
        "cc_width": tune.choice(list(range(3, 22, 2))),
        "sigma_warp": tune.uniform(0.2, 4.0),
        "sigma_grad": tune.uniform(0.2, 4.0),
    }

    points_to_evaluate = None
    if args.include_baseline:
        points_to_evaluate = [
            {
                "cc_width": 15,
                "sigma_warp": 0.5499451812055443,
                "sigma_grad": 3.6329020170516038,
            }
        ]

    search_alg = HyperOptSearch(
        metric="mean_final_dist",
        mode="min",
        random_state_seed=args.seed,
        points_to_evaluate=points_to_evaluate,
    )

    tuner = tune.Tuner(
        tune.with_resources(evaluate_config, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="mean_final_dist",
            mode="min",
            num_samples=args.num_samples,
            search_alg=search_alg,
            max_concurrent_trials=args.max_concurrent_trials,
        ),
        run_config=RunConfig(
            name="greedy_masked_cc_bayes",
            storage_path=str(output_dir),
            verbose=1,
        ),
        param_space={
            **search_space,
            "dataset": args.dataset,
            "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        },
    )

    results = tuner.fit()

    all_trials_path = output_dir / "all_trials.csv"
    results.get_dataframe().to_csv(all_trials_path, index=False)

    best = results.get_best_result(metric="mean_final_dist", mode="min")
    best_config = best.config
    best_metric = best.metrics.get("mean_final_dist")

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cc_width", "sigma_warp", "sigma_grad", "mean_final_dist"])
        writer.writerow([
            best_config["cc_width"],
            best_config["sigma_warp"],
            best_config["sigma_grad"],
            best_metric,
        ])


if __name__ == "__main__":
    main()
