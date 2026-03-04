from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import gc
import torch
import csv
import json

from mind.evotorch.data import build_pairs, split_pairs
from mind.evotorch.fireants_eval import EvalStats, GreedyRegConfig, evaluate_pairs, evaluate_pairs_stats
from mind.evotorch.model import build_model
from mind.evotorch.weights import flatten_params, set_params_from_vector
import logging
logging.getLogger("evotorch").setLevel(logging.ERROR)


def _cleanup_memory(use_gpu_eval: bool) -> None:
    gc.collect()
    if use_gpu_eval:
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class TrainConfig:
    root: Path = Path("/data/vedant/MIND")
    seed: int = 42
    test_frac: float = 0.2
    masked: bool = True
    popsize: int = 12
    num_iters: int = 40
    mutation_sigma: float = 0.05
    out_dir: Path = Path("/data/vedant/MIND/outputs/lungct")
    model_name: str = "basic"
    model_params: dict = None
    feature_mode: str = "learned"
    freeze_blocks: int = 0
    init_block1_ckpt: Path | None = None
    algo: str = "ga"
    parenthood_ratio: float = 0.25
    dataset: str = "lungct"
    dataset_root: Path | None = None
    pair_list_csv: Path | None = None
    cc_kernel_size: int | None = None
    smooth_warp_sigma: float | None = None
    smooth_grad_sigma: float | None = None
    heatmap_dice_weight: float = 0.0
    keypoint_patch_radius_vox: int = 1


def _build_model(model_name: str, params: dict | None) -> tuple[torch.nn.Module, List[dict], str]:
    return build_model(model_name, params)


def _resolve_reg_cfg_from_hparam_summary(cfg: TrainConfig) -> GreedyRegConfig:
    # Legacy defaults as fallback.
    cc = int(GreedyRegConfig.cc_kernel_size)
    sw = float(GreedyRegConfig.smooth_warp_sigma)
    sg = float(GreedyRegConfig.smooth_grad_sigma)

    summary_path = cfg.root / "benchmarks" / cfg.dataset / "hparam_search" / "summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="") as f:
            row = next(csv.DictReader(f), None)
        if row is not None:
            cc = int(row["cc_width"])
            sw = float(row["sigma_warp"])
            sg = float(row["sigma_grad"])
            print(f"[PARAMS] Loaded defaults from {summary_path}")
        else:
            print(f"[PARAMS] Empty summary at {summary_path}; using legacy defaults")
    else:
        print(f"[PARAMS] Summary not found at {summary_path}; using legacy defaults")

    if cfg.cc_kernel_size is not None:
        cc = int(cfg.cc_kernel_size)
    if cfg.smooth_warp_sigma is not None:
        sw = float(cfg.smooth_warp_sigma)
    if cfg.smooth_grad_sigma is not None:
        sg = float(cfg.smooth_grad_sigma)
    print(f"[PARAMS] cc_kernel_size={cc} smooth_warp_sigma={sw} smooth_grad_sigma={sg}")

    return GreedyRegConfig(
        loss_type="cc",
        cc_kernel_size=cc,
        smooth_warp_sigma=sw,
        smooth_grad_sigma=sg,
    )


def _fitness_factory(
    model: torch.nn.Module,
    train_pairs,
    device: str,
    masked: bool,
    reg_cfg: GreedyRegConfig,
    use_gpu_eval: bool,
    feature_mode: str,
    heatmap_dice_weight: float,
    keypoint_patch_radius_vox: int,
):
    state = {"eval_id": 0}

    def fitness(x: torch.Tensor) -> float:
        eval_device = "cuda" if use_gpu_eval else device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=eval_device, dtype=torch.float32)
        if use_gpu_eval:
            model_device = next(model.parameters()).device
            if str(model_device) != "cuda":
                model.to("cuda")
        set_params_from_vector(model, x)
        state["eval_id"] += 1
        score = evaluate_pairs(
            model,
            train_pairs,
            device=eval_device,
            masked=masked,
            reg_cfg=reg_cfg,
            feature_mode=feature_mode,
            log_prefix=f"[EVAL {state['eval_id']}] ",
            log_every=1,
            heatmap_dice_weight=heatmap_dice_weight,
            keypoint_patch_radius_vox=keypoint_patch_radius_vox,
        )
        del x
        _cleanup_memory(use_gpu_eval)
        return score

    if use_gpu_eval:
        try:
            from evotorch.decorators import on_cuda
        except Exception as exc:
            raise ImportError("evotorch.decorators.on_cuda is required for GPU eval") from exc
        return on_cuda(fitness)

    return fitness


def run_training(cfg: TrainConfig) -> None:
    try:
        from evotorch import Problem
        from evotorch.algorithms import CEM, GeneticAlgorithm
        from evotorch.operators import GaussianMutation, OnePointCrossOver
    except Exception as exc:
        raise ImportError("evotorch is required for training") from exc

    use_gpu_eval = torch.cuda.is_available()
    device = "cuda" if use_gpu_eval else "cpu"
    reg_cfg = _resolve_reg_cfg_from_hparam_summary(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.out_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                "root": str(cfg.root),
                "seed": cfg.seed,
                "test_frac": cfg.test_frac,
                "masked": cfg.masked,
                "popsize": cfg.popsize,
                "num_iters": cfg.num_iters,
                "mutation_sigma": cfg.mutation_sigma,
                "out_dir": str(cfg.out_dir),
                "model_name": cfg.model_name,
                "model_params": cfg.model_params or {},
                "feature_mode": cfg.feature_mode,
                "freeze_blocks": cfg.freeze_blocks,
                "init_block1_ckpt": str(cfg.init_block1_ckpt) if cfg.init_block1_ckpt else None,
                "algo": cfg.algo,
                "parenthood_ratio": cfg.parenthood_ratio,
                "dataset": cfg.dataset,
                "dataset_root": str(cfg.dataset_root) if cfg.dataset_root else None,
                "pair_list_csv": str(cfg.pair_list_csv) if cfg.pair_list_csv else None,
                "device": device,
                "use_gpu_eval": use_gpu_eval,
                "num_actors": "num_gpus" if use_gpu_eval else 0,
                "num_gpus_per_actor": 1 if use_gpu_eval else None,
                "model_config": str(cfg.out_dir / "model_config.json"),
                "reg_cfg": {
                    "scales": list(reg_cfg.scales),
                    "iterations": list(reg_cfg.iterations),
                    "loss_type": "cc",
                    "cc_kernel_size": reg_cfg.cc_kernel_size,
                    "smooth_warp_sigma": reg_cfg.smooth_warp_sigma,
                    "smooth_grad_sigma": reg_cfg.smooth_grad_sigma,
                },
                "heatmap_dice_weight": cfg.heatmap_dice_weight,
                "keypoint_patch_radius_vox": cfg.keypoint_patch_radius_vox,
            },
            f,
            indent=2,
        )
    pairs = build_pairs(
        cfg.root,
        dataset=cfg.dataset,
        dataset_root=cfg.dataset_root,
        pair_list_csv=cfg.pair_list_csv,
    )
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least 2 pairs after filtering, got {len(pairs)}. "
            "Increase subset size or disable --pair-list-csv."
        )
    train_pairs, test_pairs = split_pairs(pairs, seed=cfg.seed, test_frac=cfg.test_frac)

    model, model_cfgs, model_name = _build_model(cfg.model_name, cfg.model_params or {})
    model.to(device)
    model.eval()
    if cfg.init_block1_ckpt:
        state = torch.load(str(cfg.init_block1_ckpt), map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[WARN] Unexpected keys in init checkpoint: {unexpected}")
    if cfg.freeze_blocks > 0 and hasattr(model, "freeze_blocks"):
        model.freeze_blocks(cfg.freeze_blocks)
    param_vec = flatten_params(model).detach().clone()
    dim = param_vec.numel()
    del param_vec
    _cleanup_memory(use_gpu_eval)

    with (cfg.out_dir / "model_config.json").open("w") as f:
        json.dump(
            {
                "model_name": model_name,
                "model_params": cfg.model_params or {},
                "feature_mode": cfg.feature_mode,
                "freeze_blocks": cfg.freeze_blocks,
                "init_block1_ckpt": str(cfg.init_block1_ckpt) if cfg.init_block1_ckpt else None,
                "algo": cfg.algo,
                "parenthood_ratio": cfg.parenthood_ratio,
                "dataset": cfg.dataset,
                "dataset_root": str(cfg.dataset_root) if cfg.dataset_root else None,
                "pair_list_csv": str(cfg.pair_list_csv) if cfg.pair_list_csv else None,
                "heatmap_dice_weight": cfg.heatmap_dice_weight,
                "keypoint_patch_radius_vox": cfg.keypoint_patch_radius_vox,
                "model_class": model.__class__.__name__,
                "blocks": model_cfgs,
            },
            f,
            indent=2,
        )

    fitness = _fitness_factory(
        model,
        train_pairs,
        device,
        cfg.masked,
        reg_cfg,
        use_gpu_eval,
        cfg.feature_mode,
        cfg.heatmap_dice_weight,
        cfg.keypoint_patch_radius_vox,
    )

    problem = Problem(
        "min",
        fitness,
        solution_length=dim,
        initial_bounds=(-1.0, 1.0),
        device="cpu" if use_gpu_eval else device,
        num_actors="num_gpus" if use_gpu_eval else 0,
        num_gpus_per_actor=None,
    )

    if cfg.algo == "ga":
        searcher = GeneticAlgorithm(
            problem,
            popsize=cfg.popsize,
            operators=[
                OnePointCrossOver(problem, tournament_size=2),
                GaussianMutation(problem, stdev=cfg.mutation_sigma),
            ],
        )
    elif cfg.algo == "cem":
        searcher = CEM(
            problem,
            popsize=cfg.popsize,
            parenthood_ratio=cfg.parenthood_ratio,
            stdev_init=cfg.mutation_sigma,
        )
    else:
        raise ValueError(f"Unsupported algo: {cfg.algo}")

    metrics_path = cfg.out_dir / "metrics.csv"
    checkpoints_dir = cfg.out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "best_train_mean",
                "test_mean",
                "best_train_tre",
                "best_train_dice_loss",
                "test_tre",
                "test_dice_loss",
            ]
        )

        for gen in range(1, cfg.num_iters + 1):
            print(f"[GEN {gen}/{cfg.num_iters}] start")
            searcher.step()
            best = searcher.status["best"]
            best_values = best.values
            if not isinstance(best_values, torch.Tensor):
                best_values = torch.as_tensor(best_values, device=device, dtype=torch.float32)
            eval_device = "cuda" if use_gpu_eval else device
            if str(next(model.parameters()).device) != eval_device:
                model.to(eval_device)
            set_params_from_vector(model, best_values)
            train_stats: EvalStats = evaluate_pairs_stats(
                model,
                train_pairs,
                device=eval_device,
                masked=cfg.masked,
                reg_cfg=reg_cfg,
                feature_mode=cfg.feature_mode,
                heatmap_dice_weight=cfg.heatmap_dice_weight,
                keypoint_patch_radius_vox=cfg.keypoint_patch_radius_vox,
            )
            test_stats: EvalStats = evaluate_pairs_stats(
                model,
                test_pairs,
                device=eval_device,
                masked=cfg.masked,
                reg_cfg=reg_cfg,
                feature_mode=cfg.feature_mode,
                heatmap_dice_weight=cfg.heatmap_dice_weight,
                keypoint_patch_radius_vox=cfg.keypoint_patch_radius_vox,
            )
            print(
                f"[GEN {gen}/{cfg.num_iters}] best_train_mean={train_stats.mean_objective:.4f} "
                f"test_mean={test_stats.mean_objective:.4f} "
                f"train_tre={train_stats.mean_tre:.4f} test_tre={test_stats.mean_tre:.4f} "
                f"train_dice_loss={train_stats.mean_dice_loss:.4f} test_dice_loss={test_stats.mean_dice_loss:.4f}"
            )
            writer.writerow(
                [
                    gen,
                    train_stats.mean_objective,
                    test_stats.mean_objective,
                    train_stats.mean_tre,
                    train_stats.mean_dice_loss,
                    test_stats.mean_tre,
                    test_stats.mean_dice_loss,
                ]
            )
            f.flush()

            weights_path = checkpoints_dir / f"{gen:03d}.pt"
            torch.save(model.state_dict(), weights_path)
            del best_values, best
            _cleanup_memory(use_gpu_eval)

    # final evaluation on test set
    best = searcher.status["best"]
    best_values = best.values
    if not isinstance(best_values, torch.Tensor):
        best_values = torch.as_tensor(best_values, device=device, dtype=torch.float32)
    eval_device = "cuda" if use_gpu_eval else device
    if str(next(model.parameters()).device) != eval_device:
        model.to(eval_device)
    set_params_from_vector(model, best_values)
    test_stats = evaluate_pairs_stats(
        model,
        test_pairs,
        device=eval_device,
        masked=cfg.masked,
        reg_cfg=reg_cfg,
        feature_mode=cfg.feature_mode,
        heatmap_dice_weight=cfg.heatmap_dice_weight,
        keypoint_patch_radius_vox=cfg.keypoint_patch_radius_vox,
    )
    print(
        f"Test objective={test_stats.mean_objective:.6f} "
        f"(TRE={test_stats.mean_tre:.6f}, dice_loss={test_stats.mean_dice_loss:.6f})"
    )
    del best_values, best
    _cleanup_memory(use_gpu_eval)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EvoTorch training")
    parser.add_argument("--masked", action="store_true", help="Use masked CC loss")
    parser.add_argument(
        "--no-masked", action="store_true", help="Disable masked loss"
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional suffix for output folder name (e.g. run1, ablationA)",
    )
    parser.add_argument(
        "--model",
        default="basic",
        help="Model name to use: basic or res_gn_gelu",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=1,
        help="Dilation for cbam/res_gn_gelu (ignored for others)",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Number of blocks to stack (used by res_gn_gelu, cbam)",
    )
    parser.add_argument(
        "--freeze-blocks",
        type=int,
        default=0,
        help="Number of initial blocks to freeze",
    )
    parser.add_argument(
        "--init-block1-ckpt",
        type=Path,
        default=None,
        help="Checkpoint to initialize block 1 weights (e.g., best single-block)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["learned", "mind", "mind_concat"],
        default="learned",
        help="Feature source before FireANTs registration",
    )
    parser.add_argument(
        "--algo",
        choices=["ga", "cem"],
        default="ga",
        help="Search algorithm: GA (current) or CEM (Gaussian family)",
    )
    parser.add_argument(
        "--parenthood-ratio",
        type=float,
        default=0.25,
        help="CEM only: top fraction in (0,1) used as parents",
    )
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
        "--pair-list-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns case,fixed_image,moving_image to train on a subset",
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
    parser.add_argument(
        "--heatmap-dice-weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary keypoint patch Dice loss term in GA objective",
    )
    parser.add_argument(
        "--keypoint-patch-radius-vox",
        type=int,
        default=1,
        help="Radius for keypoint patch masks (1 => 3x3x3)",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.masked and args.no_masked:
        raise SystemExit("Choose only one of --masked or --no-masked")
    if args.algo == "cem" and not (0.0 < float(args.parenthood_ratio) < 1.0):
        raise SystemExit("--parenthood-ratio must be in (0, 1) when --algo cem")
    masked = True
    if args.no_masked:
        masked = False
    if args.masked:
        masked = True

    suffix = args.save.strip().replace(" ", "_")
    prefix = "masked" if masked else "unmasked"
    base = f"{prefix}_{args.model}"
    out_name = f"{base}_{suffix}" if suffix else base
    out_dir = Path(f"/data/vedant/MIND/outputs/{args.dataset}/{out_name}")
    cfg = TrainConfig(
        masked=masked,
        out_dir=out_dir,
        model_name=args.model,
        model_params={"dilation": int(args.dilation), "num_blocks": int(args.num_blocks)},
        feature_mode=args.feature_mode,
        freeze_blocks=int(args.freeze_blocks),
        init_block1_ckpt=args.init_block1_ckpt,
        algo=args.algo,
        parenthood_ratio=float(args.parenthood_ratio),
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        pair_list_csv=args.pair_list_csv,
        cc_kernel_size=args.cc_kernel_size,
        smooth_warp_sigma=args.smooth_warp_sigma,
        smooth_grad_sigma=args.smooth_grad_sigma,
        heatmap_dice_weight=float(args.heatmap_dice_weight),
        keypoint_patch_radius_vox=int(args.keypoint_patch_radius_vox),
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
