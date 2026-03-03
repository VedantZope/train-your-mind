"""Utility helpers for MIND experiments.

Preprocessing pipeline:
- mask outside the image
- normalize masked region to [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("SimpleITK is required for MIND descriptor support") from exc
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("nibabel is required for NIfTI I/O") from exc

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("SimpleITK is required for MIND descriptor support") from exc


@dataclass(frozen=True)
class PreprocessConfig:
    image_dir: str = "imagesTr"
    mask_dir: str = "masksTr"
    output_image_dir: str = "imagesTr"
    output_mask_dir: str = "masksTr"
    eps: float = 1e-8
    dtype: np.dtype = np.float32


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path
    data_root: Path
    preproc_root: Path
    benchmark_root: Path
    image_dir: Path
    keypoint_dir: Path
    mask_dir: Path
    preproc_mask_dir: Path


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def load_keypoints_csv(path: Path) -> np.ndarray:
    """Load keypoints from a CSV with no header (x,y,z per row)."""
    return np.loadtxt(str(path), delimiter=",", dtype=np.float32)


def image_from_tensor_like(source_image, tensor: torch.Tensor):
    """Create a FireANTs Image from a tensor, copying metadata from source."""
    if tensor.ndim != source_image.array.ndim:
        raise ValueError("Tensor shape must match source image batch/channel layout")
    img_tensor = tensor[0]
    channels = img_tensor.shape[0]
    is_vector = channels > 1
    if channels > 1:
        dims = img_tensor.ndim
        perm = list(range(1, dims)) + [0]
        img_tensor = img_tensor.permute(*perm)
    else:
        img_tensor = img_tensor.squeeze(0)

    np_array = img_tensor.detach().cpu().numpy()
    itk_image = sitk.GetImageFromArray(np_array, isVector=is_vector)
    src_itk = source_image.itk_image
    itk_image.SetSpacing(src_itk.GetSpacing())
    itk_image.SetDirection(src_itk.GetDirection())
    itk_image.SetOrigin(src_itk.GetOrigin())
    from fireants.io.image import Image

    return Image(itk_image, device=source_image.device)


def mind_descriptor_image_convex(source_image, radius: int = 2, dilation: int = 2):
    """Compute MIND-SSC descriptor using convex_adam_utils and return as FireANTs Image."""
    from mind.features.convex_adam_utils import MINDSSC as MINDSSC_CONVEX

    if str(source_image.device) != "cuda":
        source_image = source_image.to("cuda")
    img_tensor = source_image.array
    if img_tensor.shape[1] > 1:
        img_tensor = img_tensor[:, :1]
    mind = MINDSSC_CONVEX(img_tensor.float(), radius=radius, dilation=dilation)
    return image_from_tensor_like(source_image, mind)


def pdist_squared(x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distance for (B, D, N) tensors."""
    xx = (x ** 2).sum(dim=1, keepdim=True)
    yy = xx.transpose(2, 1)
    return xx + yy - 2 * torch.matmul(x.transpose(2, 1), x)


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    path: Path,
) -> None:
    out = nib.Nifti1Image(data, affine, header=header)
    nib.save(out, str(path))


def normalize_in_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize values inside mask to [0, 1] and zero outside."""
    if mask is None:
        vmin = float(image.min())
        vmax = float(image.max())
        denom = max(vmax - vmin, eps)
        out = (image - vmin) / denom
        return out
    else:
        mask = mask.astype(bool)

    out = np.zeros_like(image, dtype=np.float32)
    if not np.any(mask):
        return out

    masked_vals = image[mask].astype(np.float32)
    vmin = float(masked_vals.min())
    vmax = float(masked_vals.max())
    denom = max(vmax - vmin, eps)
    out[mask] = (masked_vals - vmin) / denom
    return out


def preprocess_case(
    image_path: Path,
    mask_path: Optional[Path],
    out_image_path: Path,
    out_mask_path: Optional[Path],
    config: PreprocessConfig,
) -> None:
    image, affine, header = load_nifti(image_path)
    mask = None
    if mask_path is not None and mask_path.exists():
        mask, _, _ = load_nifti(mask_path)
    norm = normalize_in_mask(image, mask, eps=config.eps).astype(config.dtype)

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(norm, affine, header, out_image_path)

    if mask is not None and out_mask_path is not None:
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        save_nifti(mask.astype(np.uint8), affine, header, out_mask_path)


def get_benchmark_paths(
    root: Path,
    data_name: str = "LungCT_L2R",
    preproc_name: str = "LungCT_L2R_preprocessed",
    benchmark_name: str = "benchmarks",
) -> BenchmarkPaths:
    """Centralize paths used across benchmarking scripts."""
    data_root = root / "data" / data_name
    preproc_root = root / "data" / preproc_name
    benchmark_root = root / benchmark_name
    image_dir = preproc_root / "imagesTr"
    keypoint_dir = data_root / "keypointsTr"
    mask_dir = data_root / "masksTr"
    preproc_mask_dir = preproc_root / "masksTr"
    return BenchmarkPaths(
        root=root,
        data_root=data_root,
        preproc_root=preproc_root,
        benchmark_root=benchmark_root,
        image_dir=image_dir,
        keypoint_dir=keypoint_dir,
        mask_dir=mask_dir,
        preproc_mask_dir=preproc_mask_dir,
    )

def compute_keypoint_loss_from_csv(
    fixed_img,
    moving_img,
    fixed_csv: Path,
    moving_csv: Path,
    space: str = "physical",
    reduction: str = "mean",
) -> float:
    """Compute mean landmark distance between fixed and moving keypoints.

    Keypoints are loaded in pixel space and compared in the requested space.
    """
    try:
        from fireants.io.keypoints import (
            BatchedKeypoints,
            Keypoints,
            compute_keypoint_distance,
        )
    except ImportError as exc:
        raise ImportError("fireants is required for keypoint loss") from exc

    fixed_pts = load_keypoints_csv(fixed_csv)
    moving_pts = load_keypoints_csv(moving_csv)

    kp_fixed = Keypoints(fixed_pts, fixed_img, device=fixed_img.device, space="pixel")
    kp_moving = Keypoints(moving_pts, moving_img, device=moving_img.device, space="pixel")

    fixed_batch = BatchedKeypoints([kp_fixed])
    moving_batch = BatchedKeypoints([kp_moving])

    dist = compute_keypoint_distance(
        fixed_batch, moving_batch, space=space, reduction=reduction
    )
    return float(dist.item())


def iter_nifti_files(path: Path) -> Iterable[Path]:
    return sorted(path.glob("*.nii")) + sorted(path.glob("*.nii.gz"))


def get_case_id_from_name(name: str) -> Optional[str]:
    """Extract case id like 'LungCT_0001' from image or keypoint filename."""
    stem = _strip_nii_suffix(name)
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-1])


def group_images_by_case(image_paths: Iterable[Path]) -> Dict[str, Dict[str, Path]]:
    """Group images by case id, keyed by suffix (e.g., '0000', '0001')."""
    grouped: Dict[str, Dict[str, Path]] = {}
    for image_path in image_paths:
        stem = _strip_nii_suffix(image_path.name)
        parts = stem.split("_")
        if len(parts) < 3:
            print(f"[SKIP] unexpected filename: {image_path.name}")
            continue
        case_id = "_".join(parts[:-1])
        suffix = parts[-1]
        grouped.setdefault(case_id, {})[suffix] = image_path
    return grouped


def iter_image_pairs(
    image_dir: Path,
    fixed_suffix: str = "0000",
    moving_suffix: str = "0001",
) -> List[Tuple[str, Path, Path]]:
    """Return (case_id, fixed_path, moving_path) for all available pairs."""
    image_paths = list(iter_nifti_files(image_dir))
    grouped = group_images_by_case(image_paths)
    pairs: List[Tuple[str, Path, Path]] = []
    for case_id, items in sorted(grouped.items()):
        print(f"Case ID: {case_id}")
        fixed_path = items.get(fixed_suffix)
        moving_path = items.get(moving_suffix)
        if fixed_path is None or moving_path is None:
            print(f"[MISSING] image pair for {case_id}")
            continue
        pairs.append((case_id, fixed_path, moving_path))
    return pairs


def keypoint_pair_paths(
    keypoint_dir: Path,
    case_id: str,
    fixed_suffix: str = "0000",
    moving_suffix: str = "0001",
) -> Tuple[Path, Path]:
    """Return keypoint CSV paths for a case id."""
    fixed_kp = keypoint_dir / f"{case_id}_{fixed_suffix}.csv"
    moving_kp = keypoint_dir / f"{case_id}_{moving_suffix}.csv"
    return fixed_kp, moving_kp


def preprocess_dataset(
    root_dir: Path,
    out_dir: Path,
    config: PreprocessConfig = PreprocessConfig(),
) -> None:
    image_dir = root_dir / config.image_dir
    mask_dir = root_dir / config.mask_dir
    out_image_dir = out_dir / config.output_image_dir
    out_mask_dir = out_dir / config.output_mask_dir

    mask_map = {
        _strip_nii_suffix(p.name): p for p in iter_nifti_files(mask_dir)
    }

    image_paths = list(iter_nifti_files(image_dir))
    total = len(image_paths)
    print(f"[INFO] Found {total} images in {image_dir}")

    for idx, image_path in enumerate(image_paths, start=1):
        key = _strip_nii_suffix(image_path.name)
        mask_path = mask_map.get(key)
        if mask_path is None:
            print(f"[MISSING] mask for {image_path.name}")
        print(f"[PROGRESS] {idx}/{total} {image_path.name}")
        out_image_path = out_image_dir / image_path.name
        out_mask_path = out_mask_dir / image_path.name
        preprocess_case(
            image_path,
            mask_path,
            out_image_path,
            out_mask_path if mask_path is not None else None,
            config,
        )


def preprocess_dataset_split(
    root_dir: Path,
    out_dir: Path,
    split: str = "Tr",
    config: PreprocessConfig = PreprocessConfig(),
) -> None:
    """Preprocess a specific split (Tr or Ts) with the same normalization pipeline."""
    split = split.strip()
    if split not in {"Tr", "Ts"}:
        raise ValueError("split must be 'Tr' or 'Ts'")
    image_dir = f"images{split}"
    mask_dir = f"masks{split}"
    cfg = PreprocessConfig(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_image_dir=image_dir,
        output_mask_dir=mask_dir,
        eps=config.eps,
        dtype=config.dtype,
    )
    preprocess_dataset(root_dir, out_dir, cfg)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LungCT_L2R dataset")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["Tr", "Ts", "both"],
        default="Tr",
        help="Which split to preprocess",
    )
    args = parser.parse_args()

    if args.split == "both":
        preprocess_dataset_split(args.root, args.out, split="Tr")
        preprocess_dataset_split(args.root, args.out, split="Ts")
    else:
        preprocess_dataset_split(args.root, args.out, split=args.split)
