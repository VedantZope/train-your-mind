from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import gc
import torch
from fireants.io.image import BatchedImages, Image
from fireants.io.imagemask import apply_mask_to_image
from fireants.io.keypoints import BatchedKeypoints, Keypoints, compute_keypoint_distance
from fireants.registration.greedy import GreedyRegistration

from mind.utils.io import image_from_tensor_like, load_keypoints_csv, mind_descriptor_image_convex
from mind.evotorch.data import PairSample
import logging

logging.getLogger("fireants").setLevel(logging.WARNING)
logging.getLogger("fireants.registration.abstract").setLevel(logging.WARNING)


@dataclass(frozen=True)
class GreedyRegConfig:
    scales: List[int] = (4, 2, 1)
    iterations: List[int] = (200, 100, 50)
    loss_type: str = "cc"
    cc_kernel_size: int = 15
    smooth_warp_sigma: float = 0.5499451812055443
    smooth_grad_sigma: float = 3.6329020170516038


def _to_feature_image(src_img: Image, feat_tensor: torch.Tensor) -> Image:
    return image_from_tensor_like(src_img, feat_tensor)


def evaluate_pairs(
    model: torch.nn.Module,
    pairs: Iterable[PairSample],
    device: str,
    masked: bool,
    reg_cfg: GreedyRegConfig,
    feature_mode: str = "learned",
    log_prefix: str = "",
    log_every: int = 0,
) -> float:
    model.eval()
    model_device = next(model.parameters()).device
    if str(model_device) != device:
        model.to(device)
    dists: List[float] = []
    pair_list = list(pairs)
    total_pairs = len(pair_list)

    for idx, sample in enumerate(pair_list, start=1):
        fixed_img = moving_img = None
        fixed_feat = moving_feat = None
        fixed_feat_img = moving_feat_img = None
        fixed_batch = moving_batch = None
        fixed_kp = moving_kp = None
        fixed_kp_batch = moving_kp_batch = None
        moved_kp_batch = None
        reg = None
        try:
            fixed_img = Image.load_file(str(sample.fixed_path), device=device)
            moving_img = Image.load_file(str(sample.moving_path), device=device)

            fixed_pts = load_keypoints_csv(sample.fixed_kp)
            moving_pts = load_keypoints_csv(sample.moving_kp)

            if feature_mode == "learned":
                with torch.inference_mode():
                    fixed_feat = model(fixed_img.array.float())
                    moving_feat = model(moving_img.array.float())
            elif feature_mode == "mind":
                fixed_feat = mind_descriptor_image_convex(fixed_img).array.float()
                moving_feat = mind_descriptor_image_convex(moving_img).array.float()
            elif feature_mode == "mind_concat":
                with torch.inference_mode():
                    fixed_learned = model(fixed_img.array.float())
                    moving_learned = model(moving_img.array.float())
                fixed_mind = mind_descriptor_image_convex(fixed_img).array.float()
                moving_mind = mind_descriptor_image_convex(moving_img).array.float()
                fixed_feat = torch.cat([fixed_learned, fixed_mind], dim=1)
                moving_feat = torch.cat([moving_learned, moving_mind], dim=1)
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode}")

            fixed_feat_img = _to_feature_image(fixed_img, fixed_feat)
            moving_feat_img = _to_feature_image(moving_img, moving_feat)

            if masked:
                fixed_mask = Image.load_file(str(sample.fixed_mask), device=device)
                moving_mask = Image.load_file(str(sample.moving_mask), device=device)
                fixed_feat_img = apply_mask_to_image(fixed_feat_img, fixed_mask)
                moving_feat_img = apply_mask_to_image(moving_feat_img, moving_mask)
                loss_type = "masked_cc"
            else:
                loss_type = "cc"

            fixed_batch = BatchedImages([fixed_feat_img])
            moving_batch = BatchedImages([moving_feat_img])

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
                max_tolerance_iters=1000,
                progress_bar=False,
            )
            reg.optimize()

            moved_kp_batch = reg.evaluate_keypoints(fixed_kp_batch, moving_kp_batch)
            final_dist = compute_keypoint_distance(
                moved_kp_batch, moving_kp_batch, space="physical", reduction="mean"
            ).item()
            dists.append(final_dist)
            if log_every > 0 and idx % log_every == 0:
                print(f"{log_prefix}pair {idx}/{total_pairs} dist={final_dist:.4f}")
        finally:
            del moved_kp_batch
            del reg
            del fixed_kp_batch, moving_kp_batch
            del fixed_kp, moving_kp
            del fixed_batch, moving_batch
            del fixed_feat_img, moving_feat_img
            del fixed_feat, moving_feat
            del fixed_img, moving_img
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    return float(sum(dists) / max(1, len(dists)))
