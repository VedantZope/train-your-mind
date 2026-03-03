from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import json
import numpy as np
import torch
from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image
from fireants.io.keypoints import BatchedKeypoints, Keypoints, compute_keypoint_distance
from fireants.registration.greedy import GreedyRegistration
import nibabel as nib

from mind.evotorch.model import build_model
from mind.evotorch.fireants_eval import GreedyRegConfig
from mind.utils.io import (
    image_from_tensor_like,
    iter_image_pairs,
    keypoint_pair_paths,
    load_keypoints_csv,
    mind_descriptor_image_convex,
    save_nifti,
    normalize_in_mask,
)


@dataclass(frozen=True)
class PairSample:
    case_id: str
    fixed_path: Path
    moving_path: Path
    fixed_kp: Path
    moving_kp: Path
    fixed_mask: Optional[Path]
    moving_mask: Optional[Path]


def _build_pairs(data_root: Path, split: str, masked: bool) -> List[PairSample]:
    if split == "train":
        image_dir = data_root / "imagesTr"
        keypoint_dir = data_root / "keypointsTr"
        mask_dir = data_root / "masksTr"
    else:
        image_dir = data_root / "imagesTs"
        keypoint_dir = data_root / "keypointsTs"
        mask_dir = data_root / "masksTs"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {image_dir}")
    if not keypoint_dir.exists():
        raise FileNotFoundError(f"Missing keypoint dir: {keypoint_dir}")
    if masked and not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask dir for masked eval: {mask_dir}")

    pairs = iter_image_pairs(image_dir)
    samples: List[PairSample] = []
    for case_id, fixed_path, moving_path in pairs:
        fixed_kp, moving_kp = keypoint_pair_paths(keypoint_dir, case_id)
        fixed_mask = mask_dir / fixed_path.name if masked else None
        moving_mask = mask_dir / moving_path.name if masked else None
        samples.append(
            PairSample(
                case_id=case_id,
                fixed_path=fixed_path,
                moving_path=moving_path,
                fixed_kp=fixed_kp,
                moving_kp=moving_kp,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
            )
        )
    return samples


def _case_suffix_from_path(path: Path) -> str:
    stem = path.name
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        stem = stem[: -len(".nii")]
    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    return parts[-2]


def _maybe_single_channel(t: torch.Tensor) -> torch.Tensor:
    if t.shape[1] > 1:
        return t[:, :1]
    return t


def _feature_image_identity(src_img: Image) -> Image:
    return src_img


def _feature_image_mind(src_img: Image) -> Image:
    return mind_descriptor_image_convex(src_img)


def _feature_image_model(model: torch.nn.Module, src_img: Image, device: str) -> Image:
    model.eval()
    if str(next(model.parameters()).device) != device:
        model.to(device)
    with torch.no_grad():
        tensor = _maybe_single_channel(src_img.array.float())
        feat = model(tensor)
    return image_from_tensor_like(src_img, feat)


def _export_feature_image(img: Image, out_path: Path) -> None:
    """Normalize to [0, 1], scale by 2**16, cast to int16, and save as NIfTI.

    Handles both single-channel and multi-channel feature images by placing channels
    in the last dimension.
    """
    tensor = img.array.detach().cpu()
    if tensor.ndim < 4:
        raise ValueError(f"Unexpected tensor shape for export: {tensor.shape}")

    # Drop batch dimension, keep (C, D, H, W, ...)
    tensor = tensor[0]
    channels = tensor.shape[0]
    if channels > 1:
        dims = tensor.ndim
        perm = list(range(1, dims)) + [0]
        tensor = tensor.permute(*perm)
    else:
        tensor = tensor.squeeze(0)

    data = tensor.numpy().astype(np.float32)
    vmin = float(data.min())
    vmax = float(data.max())
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin)
    else:
        data = np.zeros_like(data, dtype=np.float32)

    # data = np.clip(data, 0.0, 1.0)
    data = normalize_in_mask(data, None)
    data = (data * (2**12)).astype(np.int16)

    affine = np.eye(4, dtype=np.float32)
    header = nib.Nifti1Header()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(data, affine, header, out_path)


def evaluate_pairs(
    pairs: Iterable[PairSample],
    device: str,
    masked: bool,
    reg_cfg: GreedyRegConfig,
    model: Optional[torch.nn.Module] = None,
    model_kind: str = "identity",
    log_every: int = 0,
    save_displacement: bool = False,
    results_dir: Optional[Path] = None,
    save_images: bool = False,
) -> float:
    dists: List[float] = []
    pair_list = list(pairs)
    total_pairs = len(pair_list)
    if save_displacement:
        if results_dir is None:
            raise ValueError("results_dir is required when save_displacement=True")
        results_dir.mkdir(parents=True, exist_ok=True)

    images_root: Optional[Path] = None
    if save_images:
        images_root = Path("images") / model_kind
        images_root.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(pair_list, start=1):
        has_kp = sample.fixed_kp.exists() and sample.moving_kp.exists()
        if not has_kp:
            print(f"[WARNING] Missing keypoints for {sample.case_id}, skipping TRE")
        fixed_img = Image.load_file(str(sample.fixed_path), device=device)
        moving_img = Image.load_file(str(sample.moving_path), device=device)

        if model_kind == "mind":
            fixed_feat_img = _feature_image_mind(fixed_img)
            moving_feat_img = _feature_image_mind(moving_img)
        elif model_kind == "model":
            if model is None:
                raise ValueError("model is required when model_kind='model'")
            fixed_feat_img = _feature_image_model(model, fixed_img, device)
            moving_feat_img = _feature_image_model(model, moving_img, device)
        else:
            fixed_feat_img = _feature_image_identity(fixed_img)
            moving_feat_img = _feature_image_identity(moving_img)

        if masked:
            if sample.fixed_mask is None or sample.moving_mask is None:
                raise ValueError("Masked eval requires mask paths for each sample")
            fixed_mask = Image.load_file(str(sample.fixed_mask), device=device)
            moving_mask = Image.load_file(str(sample.moving_mask), device=device)
            fixed_feat_img = apply_mask_to_image(fixed_feat_img, fixed_mask)
            moving_feat_img = apply_mask_to_image(moving_feat_img, moving_mask)
            loss_type = f"masked_{reg_cfg.loss_type}"
        else:
            loss_type = reg_cfg.loss_type

        if save_images and images_root is not None:
            fixed_out = images_root / f"{sample.case_id}_fixed.nii.gz"
            moving_out = images_root / f"{sample.case_id}_moving.nii.gz"
            _export_feature_image(fixed_feat_img, fixed_out)
            _export_feature_image(moving_feat_img, moving_out)

        fixed_batch = BatchedImages([fixed_feat_img])
        moving_batch = BatchedImages([moving_feat_img])

        if has_kp:
            fixed_pts = load_keypoints_csv(sample.fixed_kp)
            moving_pts = load_keypoints_csv(sample.moving_kp)

            fixed_kp = Keypoints(fixed_pts, fixed_img, device=device, space="pixel")
            moving_kp = Keypoints(moving_pts, moving_img, device=device, space="pixel")
            fixed_kp_batch = BatchedKeypoints([fixed_kp])
            moving_kp_batch = BatchedKeypoints([moving_kp])

        reg = GreedyRegistration(
            scales=list(reg_cfg.scales),
            iterations=list(reg_cfg.iterations),
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            loss_type=loss_type,
            cc_kernel_size=int(reg_cfg.cc_kernel_size),
            smooth_warp_sigma=float(reg_cfg.smooth_warp_sigma),
            smooth_grad_sigma=float(reg_cfg.smooth_grad_sigma),
            progress_bar=False,
        )
        reg.optimize()
        if save_displacement:
            fixed_id = _case_suffix_from_path(sample.fixed_path)
            moving_id = _case_suffix_from_path(sample.moving_path)
            out_name = f"disp_{fixed_id}_{moving_id}.npz"
            reg.save_as_scipy_transforms(str(results_dir / out_name))

        if has_kp:
            moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
            final_dist = compute_keypoint_distance(
                moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
            ).item()
            dists.append(final_dist)
            if log_every > 0 and idx % log_every == 0:
                print(f"pair {idx}/{total_pairs} dist={final_dist:.4f}")

    return float(sum(dists) / max(1, len(dists))), dists


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


def run_inference(
    weights: Path,
    split: str,
    masked: bool,
    model_kind: str,
    data_root: Path,
    save_displacement: bool,
    save_images: bool,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg_cfg = GreedyRegConfig(loss_type="cc")

    model = None
    if model_kind == "model":
        model = _load_model_from_weights(weights)

    pairs = _build_pairs(data_root, split, masked)
    mean_dist, all_dists = evaluate_pairs(
        pairs,
        device=device,
        masked=masked,
        reg_cfg=reg_cfg,
        model=model,
        model_kind=model_kind,
        log_every=1,
        save_displacement=save_displacement,
        results_dir=Path("/data/vedant/MIND/infer/results") if save_displacement else None,
        save_images=save_images,
    )
    print(f"{split} mean TRE: {mean_dist:.6f}")
    if split == "train":
        import matplotlib.pyplot as plt
        import numpy as np

        print(f"Size of all_dists array: {len(all_dists)}")
        first_16 = all_dists[:16]
        last_4 = all_dists[-4:]
        mean_first_16 = float(np.mean(first_16)) if first_16 else float('nan')
        mean_last_4 = float(np.mean(last_4)) if last_4 else float('nan')

        print(f"Mean of first 16: {mean_first_16:.6f}")
        print(f"Mean of last 4: {mean_last_4:.6f}")
        
        # Plot the means for first 16 and last 4
        plt.figure(figsize=(8, 4))
        plt.bar(['First 16', 'Last 4'], [mean_first_16, mean_last_4], color=['blue', 'orange'])
        plt.ylabel('Mean TRE')
        plt.title('Mean TRE for first 16 and last 4 (Train Split)')
        plt.show()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inference: model/MIND/identity + FireANTs TRE")
    parser.add_argument("--weights", type=Path, required=False, help="Path to model checkpoint (.pt)")
    parser.add_argument("--train", action="store_true", help="Evaluate on train split")
    parser.add_argument("--test", action="store_true", help="Evaluate on test split")
    parser.add_argument("--masked", action="store_true", help="Use masked CC loss")
    parser.add_argument("--no-masked", action="store_true", help="Disable masked loss")
    parser.add_argument(
        "--model",
        default="auto",
        help="auto (use weights), mind, identity",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/vedant/MIND/data/LungCT_L2R_preprocessed"),
        help="Dataset root containing imagesTr/imagesTs and keypointsTr/keypointsTs",
    )
    parser.add_argument(
        "--save-displacement",
        action="store_true",
        help="Save warps in Learn2Reg format via reg.save_as_scipy_transforms",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save normalized feature images as NIfTI under images/<model>/",
    )
    args = parser.parse_args()

    if args.train == args.test:
        raise SystemExit("Choose exactly one of --train or --test")
    if args.masked and args.no_masked:
        raise SystemExit("Choose only one of --masked or --no-masked")

    masked = False
    if args.masked:
        masked = True
    print(f"Masked: {masked}")

    model_kind = args.model.strip().lower()
    if model_kind == "auto":
        model_kind = "model"
    elif model_kind not in {"model", "mind", "identity"}:
        raise SystemExit("--model must be one of: auto, mind, identity")

    if model_kind == "model" and args.weights is None:
        raise SystemExit("--weights is required when using model/auto")

    split = "train" if args.train else "test"
    run_inference(
        weights=args.weights or Path(""),
        split=split,
        masked=masked,
        model_kind=model_kind,
        data_root=args.data_root,
        save_displacement=args.save_displacement,
        save_images=args.save_images,
    )


if __name__ == "__main__":
    main()
