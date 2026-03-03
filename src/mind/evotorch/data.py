from __future__ import annotations

import csv
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
    pair_list_csv: Path | None = None,
) -> List[PairSample]:
    def _load_subset_rows(path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
        if not path.exists():
            raise FileNotFoundError(f"Pair list CSV not found: {path}")
        pair_keys: set[tuple[str, str]] = set()
        triplet_keys: set[tuple[str, str, str]] = set()
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"case", "fixed_image", "moving_image"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{path} must contain header columns: case,fixed_image,moving_image"
                )
            for row in reader:
                case = (row.get("case") or "").strip()
                fixed = (row.get("fixed_image") or "").strip()
                moving = (row.get("moving_image") or "").strip()
                if not case or not fixed or not moving:
                    continue
                pair_keys.add((fixed, moving))
                triplet_keys.add((case, fixed, moving))
        if not pair_keys:
            raise ValueError(f"No valid rows found in pair list CSV: {path}")
        return pair_keys, triplet_keys

    image_dir, keypoint_dir, mask_dir = _resolve_pair_dirs(
        root=root, dataset=dataset, dataset_root=dataset_root
    )
    subset_pairs: set[tuple[str, str]] | None = None
    subset_triplets: set[tuple[str, str, str]] | None = None
    if pair_list_csv is not None:
        subset_pairs, subset_triplets = _load_subset_rows(pair_list_csv)
        samples: List[PairSample] = []
        missing_pairs: list[tuple[str, str]] = []

        def _infer_case_id(name: str) -> str:
            stem = name
            if stem.endswith(".nii.gz"):
                stem = stem[: -len(".nii.gz")]
            elif stem.endswith(".nii"):
                stem = stem[: -len(".nii")]
            parts = stem.split("_")
            if len(parts) < 3:
                return stem
            return "_".join(parts[:-1])

        for fixed_name, moving_name in sorted(subset_pairs):
            fixed_path = image_dir / fixed_name
            moving_path = image_dir / moving_name
            if not fixed_path.exists() or not moving_path.exists():
                missing_pairs.append((fixed_name, moving_name))
                continue

            case_id = None
            for c, f, m in subset_triplets:
                if f == fixed_name and m == moving_name:
                    case_id = c
                    break
            if not case_id:
                case_id = _infer_case_id(fixed_name)

            fixed_kp, moving_kp = keypoint_pair_paths(keypoint_dir, case_id)
            fixed_mask = mask_dir / fixed_name
            moving_mask = mask_dir / moving_name
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

        if missing_pairs:
            preview = ", ".join([f"{a}|{b}" for a, b in missing_pairs[:5]])
            raise ValueError(
                f"{len(missing_pairs)} requested pairs were not found in {image_dir}. "
                f"Examples: {preview}"
            )
        return samples

    pairs = iter_image_pairs(image_dir)
    samples: List[PairSample] = []
    found_pairs: set[tuple[str, str]] = set()
    found_triplets: set[tuple[str, str, str]] = set()
    for case_id, fixed_path, moving_path in pairs:
        fixed_name = fixed_path.name
        moving_name = moving_path.name
        if subset_pairs is not None:
            pair_key = (fixed_name, moving_name)
            triplet_key = (case_id, fixed_name, moving_name)
            if pair_key not in subset_pairs and triplet_key not in subset_triplets:
                continue
            found_pairs.add(pair_key)
            found_triplets.add(triplet_key)

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

    if subset_pairs is not None:
        missing_pairs = sorted(subset_pairs - found_pairs)
        if missing_pairs:
            preview = ", ".join([f"{a}|{b}" for a, b in missing_pairs[:5]])
            raise ValueError(
                f"{len(missing_pairs)} requested pairs were not found in {image_dir}. "
                f"Examples: {preview}"
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
