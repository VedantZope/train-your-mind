from __future__ import annotations

from pathlib import Path

import numpy as np

from mind.utils.io import load_keypoints_csv, load_nifti, save_nifti


def keypoints_to_patch_mask(
    shape: tuple[int, ...],
    keypoints_xyz: np.ndarray,
    radius_vox: int = 1,
) -> np.ndarray:
    """Create a binary mask with cubic patches around keypoints.

    Each keypoint becomes a (2*radius_vox+1)^3 patch of ones in voxel space.
    Coordinates are assumed to be voxel-index coordinates in (x, y, z) order.
    """
    if len(shape) < 3:
        raise ValueError(f"Expected at least 3D shape, got: {shape}")
    if radius_vox < 0:
        raise ValueError("radius_vox must be >= 0")

    mask = np.zeros(shape[:3], dtype=np.uint8)
    if keypoints_xyz.size == 0:
        return mask

    # Round to nearest voxel center, then paint a local cube.
    pts = np.rint(keypoints_xyz[:, :3]).astype(np.int64)
    sx, sy, sz = mask.shape
    for x, y, z in pts:
        x0 = max(0, x - radius_vox)
        x1 = min(sx - 1, x + radius_vox)
        y0 = max(0, y - radius_vox)
        y1 = min(sy - 1, y + radius_vox)
        z0 = max(0, z - radius_vox)
        z1 = min(sz - 1, z + radius_vox)
        mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = 1
    return mask


def generate_keypoint_patch_mask(
    image_path: Path,
    keypoint_csv: Path,
    out_path: Path,
    radius_vox: int = 1,
) -> None:
    """Generate and save a keypoint patch mask aligned to a reference image."""
    image, affine, header = load_nifti(image_path)
    keypoints = load_keypoints_csv(keypoint_csv)
    mask = keypoints_to_patch_mask(tuple(image.shape), keypoints, radius_vox=radius_vox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(mask, affine, header, out_path)
