from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str
    t1_file: str
    t1_key: str
    t2_file: str
    t2_key: str
    label_file: str
    label_key: str
    changed_value: int
    unchanged_value: int


DATASET_SPECS: dict[str, DatasetSpec] = {
    "barbara": DatasetSpec(
        name="Barbara",
        folder="Barbara",
        t1_file="barbara_2013.mat",
        t1_key="HypeRvieW",
        t2_file="barbara_2014.mat",
        t2_key="HypeRvieW",
        label_file="barbara_gtChanges.mat",
        label_key="HypeRvieW",
        changed_value=1,
        unchanged_value=2,
    ),
    "bayarea": DatasetSpec(
        name="BayArea",
        folder="BayArea",
        t1_file="Bay_Area_2013.mat",
        t1_key="HypeRvieW",
        t2_file="Bay_Area_2015.mat",
        t2_key="HypeRvieW",
        label_file="bayArea_gtChanges.mat",
        label_key="HypeRvieW",
        changed_value=1,
        unchanged_value=2,
    ),
    "farmland": DatasetSpec(
        name="Farmland",
        folder="Farmland",
        t1_file="farm06.mat",
        t1_key="imgh",
        t2_file="farm07.mat",
        t2_key="imghl",
        label_file="label.mat",
        label_key="label",
        changed_value=1,
        unchanged_value=0,
    ),
    "river": DatasetSpec(
        name="River",
        folder="river_dataset",
        t1_file="river_before.mat",
        t1_key="river_before",
        t2_file="river_after.mat",
        t2_key="river_after",
        label_file="groundtruth.mat",
        label_key="lakelabel_v1",
        changed_value=255,
        unchanged_value=0,
    ),
    "china": DatasetSpec(
        name="China",
        folder="China_change_detection",
        t1_file="China_Change_Dataset.mat",
        t1_key="T1",
        t2_file="China_Change_Dataset.mat",
        t2_key="T2",
        label_file="China_Change_Dataset.mat",
        label_key="Binary",
        changed_value=1,
        unchanged_value=0,
    ),
}


def available_datasets() -> list[str]:
    return [spec.name for spec in DATASET_SPECS.values()]


def canonical_dataset_name(name: str) -> str:
    key = name.lower().replace("_", "").replace("-", "")
    if key not in DATASET_SPECS:
        choices = ", ".join(available_datasets())
        raise ValueError(f"Unknown dataset {name!r}. Choose one of: {choices}.")
    return key


def _load_mat_array(path: Path, key: str) -> np.ndarray:
    mat = loadmat(path)
    if key not in mat:
        visible_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"{path} does not contain {key!r}; keys: {visible_keys}")
    return mat[key]


def load_dataset_pair(data_root: str | Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetSpec]:
    spec = DATASET_SPECS[canonical_dataset_name(name)]
    root = Path(data_root) / spec.folder
    t1 = _load_mat_array(root / spec.t1_file, spec.t1_key).astype(np.float32, copy=False)
    t2 = _load_mat_array(root / spec.t2_file, spec.t2_key).astype(np.float32, copy=False)
    label = _load_mat_array(root / spec.label_file, spec.label_key)
    if t1.shape != t2.shape:
        raise ValueError(f"{spec.name} T1/T2 shape mismatch: {t1.shape} vs {t2.shape}")
    if t1.shape[:2] != label.shape:
        raise ValueError(f"{spec.name} image/label shape mismatch: {t1.shape[:2]} vs {label.shape}")
    _normalize_pair_inplace(t1, t2)
    return t1, t2, label, spec


def _normalize_pair_inplace(t1: np.ndarray, t2: np.ndarray) -> None:
    for band in range(t1.shape[2]):
        b1 = t1[:, :, band]
        b2 = t2[:, :, band]
        low = min(float(np.min(b1)), float(np.min(b2)))
        high = max(float(np.max(b1)), float(np.max(b2)))
        scale = high - low
        if scale <= 1e-12:
            t1[:, :, band] = 0.0
            t2[:, :, band] = 0.0
        else:
            t1[:, :, band] = (b1 - low) / scale
            t2[:, :, band] = (b2 - low) / scale


def _flat_to_positions(flat_indices: np.ndarray, width: int) -> np.ndarray:
    rows = flat_indices // width
    cols = flat_indices % width
    return np.stack([rows, cols], axis=1).astype(np.int64, copy=False)


def _sample_indices(
    rng: np.random.Generator,
    indices: np.ndarray,
    count: int,
    *,
    replace: bool = False,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if replace:
        return rng.choice(indices, size=count, replace=True)
    count = min(count, len(indices))
    return rng.choice(indices, size=count, replace=False)


def build_splits(
    label: np.ndarray,
    spec: DatasetSpec,
    train_number: int,
    seed: int,
    max_test_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat_label = label.reshape(-1)
    changed_all = np.flatnonzero(flat_label == spec.changed_value)
    unchanged_all = np.flatnonzero(flat_label == spec.unchanged_value)
    if len(changed_all) == 0 or len(unchanged_all) == 0:
        raise ValueError(f"{spec.name} needs both changed and unchanged labels.")

    train_each = min(train_number, len(changed_all), len(unchanged_all))
    changed_train = _sample_indices(rng, changed_all, train_each)
    unchanged_train = _sample_indices(rng, unchanged_all, train_each)

    changed_test = np.setdiff1d(changed_all, changed_train, assume_unique=False)
    unchanged_test = np.setdiff1d(unchanged_all, unchanged_train, assume_unique=False)

    if max_test_points is not None and max_test_points > 0:
        each = max_test_points // 2
        changed_test = _sample_indices(rng, changed_test, each)
        unchanged_test = _sample_indices(rng, unchanged_test, max_test_points - len(changed_test))

    train_flat = np.concatenate([unchanged_train, changed_train])
    train_y = np.concatenate(
        [
            np.zeros(len(unchanged_train), dtype=np.int64),
            np.ones(len(changed_train), dtype=np.int64),
        ]
    )
    test_flat = np.concatenate([unchanged_test, changed_test])
    test_y = np.concatenate(
        [
            np.zeros(len(unchanged_test), dtype=np.int64),
            np.ones(len(changed_test), dtype=np.int64),
        ]
    )

    train_order = rng.permutation(len(train_flat))
    test_order = rng.permutation(len(test_flat))
    width = label.shape[1]
    return (
        _flat_to_positions(train_flat[train_order], width),
        train_y[train_order],
        _flat_to_positions(test_flat[test_order], width),
        test_y[test_order],
    )


class HSIPatchPairDataset(Dataset):
    def __init__(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        positions: np.ndarray,
        labels: Iterable[int] | np.ndarray,
        patch_size: int,
    ) -> None:
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd so each sample has a center pixel.")
        self.patch_size = patch_size
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        pad = patch_size // 2
        if pad > 0:
            self.t1 = np.pad(t1, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            self.t2 = np.pad(t2, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        else:
            self.t1 = t1
            self.t2 = t2

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = self.positions[index]
        p = self.patch_size
        patch_t1 = self.t1[row : row + p, col : col + p, :]
        patch_t2 = self.t2[row : row + p, col : col + p, :]
        x1 = np.ascontiguousarray(patch_t1.transpose(2, 0, 1).reshape(patch_t1.shape[2], p * p))
        x2 = np.ascontiguousarray(patch_t2.transpose(2, 0, 1).reshape(patch_t2.shape[2], p * p))
        y = np.array(self.labels[index], dtype=np.int64)
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(y)
