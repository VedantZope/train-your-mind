from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from mind.utils.io import get_benchmark_paths, iter_image_pairs, keypoint_pair_paths


@dataclass(frozen=True)
class PairSample:
    case_id: str
    fixed_path: Path
    moving_path: Path
    fixed_kp: Path
    moving_kp: Path
    fixed_mask: Path
    moving_mask: Path


def _resolve_pair_dirs(
    root: Path,
    dataset: str = "lungct",
    dataset_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    ds = dataset.lower()
    if ds == "lungct":
        paths = get_benchmark_paths(root)
        return paths.image_dir, paths.keypoint_dir, paths.preproc_mask_dir
    if ds == "nlst":
        base = dataset_root or Path("/mnt/rohit_data2/NLST/NLST")
        image_dir = base / "imagesProcessedTr"
        if not image_dir.exists():
            image_dir = base / "imagesTr"
        keypoint_dir = base / "keypointsTr"
        mask_dir = base / "masksTr"
        return image_dir, keypoint_dir, mask_dir
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_pairs(
    root: Path,
    dataset: str = "lungct",
    dataset_root: Path | None = None,
) -> List[PairSample]:
    image_dir, keypoint_dir, mask_dir = _resolve_pair_dirs(
        root=root, dataset=dataset, dataset_root=dataset_root
    )
    pairs = iter_image_pairs(image_dir)
    samples: List[PairSample] = []
    for case_id, fixed_path, moving_path in pairs:
        fixed_kp, moving_kp = keypoint_pair_paths(keypoint_dir, case_id)
        fixed_mask = mask_dir / fixed_path.name
        moving_mask = mask_dir / moving_path.name
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


def split_pairs(
    pairs: Sequence[PairSample], seed: int = 42, test_frac: float = 0.2
) -> Tuple[List[PairSample], List[PairSample]]:
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    pairs = list(pairs)
    rng = __import__("random").Random(seed)
    rng.shuffle(pairs)
    n_test = max(1, int(round(len(pairs) * test_frac)))
    test = pairs[:n_test]
    train = pairs[n_test:]
    return train, test
