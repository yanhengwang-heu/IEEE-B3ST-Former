from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import HSIPatchPairDataset, available_datasets, build_splits, canonical_dataset_name, load_dataset_pair
from .metrics import binary_metrics
from .models import FixedBandSSTACIDetector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _device(force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _move_batch(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1, x2, y = batch
    return x1.to(device, non_blocking=True), x2.to(device, non_blocking=True), y.to(device, non_blocking=True)


def fisher_band_selection(
    t1: np.ndarray,
    t2: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    keep_bands: int,
    mode: str = "diverse",
) -> np.ndarray:
    rows = positions[:, 0]
    cols = positions[:, 1]
    diff = np.abs(t2[rows, cols, :] - t1[rows, cols, :]).astype(np.float64, copy=False)
    changed = diff[labels == 1]
    unchanged = diff[labels == 0]
    if len(changed) == 0 or len(unchanged) == 0:
        raise ValueError("Fisher band selection needs both changed and unchanged training samples.")
    mean_gap = changed.mean(axis=0) - unchanged.mean(axis=0)
    denom = changed.var(axis=0) + unchanged.var(axis=0) + 1e-8
    scores = (mean_gap * mean_gap) / denom
    if mode == "topk":
        top = np.argpartition(-scores, kth=min(keep_bands, scores.shape[0]) - 1)[:keep_bands]
    elif mode == "diverse":
        groups = np.array_split(np.arange(scores.shape[0]), keep_bands)
        top = np.array([group[np.argmax(scores[group])] for group in groups if len(group) > 0])
        if len(top) < keep_bands:
            rest = np.setdiff1d(np.arange(scores.shape[0]), top, assume_unique=False)
            fill = rest[np.argpartition(-scores[rest], kth=keep_bands - len(top) - 1)[: keep_bands - len(top)]]
            top = np.concatenate([top, fill])
    else:
        raise ValueError(f"Unknown band selection mode: {mode}")
    return np.sort(top).astype(np.int64, copy=False)


def _select_top_bands(scores: np.ndarray, keep_bands: int, mode: str) -> np.ndarray:
    keep_bands = min(int(keep_bands), int(scores.shape[0]))
    if mode == "topk":
        top = np.argpartition(-scores, kth=keep_bands - 1)[:keep_bands]
    elif mode == "diverse":
        groups = np.array_split(np.arange(scores.shape[0]), keep_bands)
        top = np.array([group[np.argmax(scores[group])] for group in groups if len(group) > 0])
        if len(top) < keep_bands:
            rest = np.setdiff1d(np.arange(scores.shape[0]), top, assume_unique=False)
            fill = rest[np.argpartition(-scores[rest], kth=keep_bands - len(top) - 1)[: keep_bands - len(top)]]
            top = np.concatenate([top, fill])
    else:
        raise ValueError(f"Unknown fixed band ranking mode: {mode}")
    return np.sort(top).astype(np.int64, copy=False)


def graph_guided_band_selection(
    t1: np.ndarray,
    t2: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    keep_bands: int,
    mode: str = "diverse",
    graph_neighbors: int = 8,
    graph_weight: float = 0.35,
) -> np.ndarray:
    rows = positions[:, 0]
    cols = positions[:, 1]
    diff = np.abs(t2[rows, cols, :] - t1[rows, cols, :]).astype(np.float64, copy=False)
    changed = diff[labels == 1]
    unchanged = diff[labels == 0]
    if len(changed) == 0 or len(unchanged) == 0:
        raise ValueError("Graph-guided band selection needs both changed and unchanged training samples.")

    mean_gap = changed.mean(axis=0) - unchanged.mean(axis=0)
    denom = changed.var(axis=0) + unchanged.var(axis=0) + 1e-8
    fisher_scores = (mean_gap * mean_gap) / denom

    diff_centered = diff - diff.mean(axis=0, keepdims=True)
    corr = np.corrcoef(diff_centered, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    neighbors = min(max(int(graph_neighbors), 1), max(corr.shape[0] - 1, 1))
    if neighbors < corr.shape[0]:
        kth = corr.shape[0] - neighbors - 1
        threshold = np.partition(corr, kth=kth, axis=1)[:, kth][:, None]
        graph = np.where(corr >= threshold, corr, 0.0)
    else:
        graph = corr
    graph = np.maximum(graph, graph.T)
    degree = graph.sum(axis=1)
    degree_norm = degree / (degree.max() + 1e-8)
    fisher_norm = fisher_scores / (fisher_scores.max() + 1e-8)
    scores = fisher_norm + float(graph_weight) * degree_norm
    return _select_top_bands(scores, keep_bands, mode)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch in loader:
        x1, x2, y = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * y.numel()
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        seen += y.numel()
    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | list[list[int]]]:
    model.eval()
    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for batch in loader:
        x1, x2, y = _move_batch(batch, device)
        logits = model(x1, x2)
        targets.append(y.detach().cpu().numpy())
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return binary_metrics(np.concatenate(targets), np.concatenate(preds))


@torch.no_grad()
def summarize_selected_bands(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 4) -> list[int]:
    model.eval()
    counts: dict[int, int] = {}
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x1, x2, _ = _move_batch(batch, device)
        _, info = model(x1, x2, return_info=True)
        if "indices" not in info:
            return []
        for band in info["indices"].detach().cpu().reshape(-1).tolist():
            counts[int(band)] = counts.get(int(band), 0) + 1
    return [band for band, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def run_dataset(args: argparse.Namespace, dataset_name: str) -> dict[str, object]:
    seed_everything(args.seed)
    device = _device(args.cpu)
    t0 = time.time()

    t1, t2, label, spec = load_dataset_pair(args.data_root, dataset_name)
    train_pos, train_y, test_pos, test_y = build_splits(
        label=label,
        spec=spec,
        train_number=args.train_number,
        seed=args.seed,
        max_test_points=args.max_test_points,
    )
    train_set = HSIPatchPairDataset(t1, t2, train_pos, train_y, args.patch_size)
    test_set = HSIPatchPairDataset(t1, t2, test_pos, test_y, args.patch_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    input_bands = t1.shape[2]
    selected_bands = graph_guided_band_selection(
        t1=t1,
        t2=t2,
        positions=train_pos,
        labels=train_y,
        keep_bands=min(args.keep_bands, input_bands),
        mode=args.fixed_band_ranking,
        graph_neighbors=args.graph_neighbors,
        graph_weight=args.graph_weight,
    )
    print(
        f"[{spec.name}] selected bands: "
        f"{selected_bands[: min(20, len(selected_bands))].tolist()}"
        f"{' ...' if len(selected_bands) > 20 else ''}",
        flush=True,
    )
    model = FixedBandSSTACIDetector(
        input_bands=input_bands,
        patch_size=args.patch_size,
        selected_bands=selected_bands,
        dim=args.sst_dim,
        depth=args.sst_depth,
        heads=args.sst_heads,
        dim_head=args.sst_dim_head,
        mlp_dim=args.sst_mlp_dim,
        b_dim=args.b_dim,
        b_depth=args.b_depth,
        b_heads=args.b_heads,
        b_dim_head=args.b_dim_head,
        b_mlp_dim=args.b_mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_acc": train_acc,
        }
        print(
            f"[{spec.name}] epoch {epoch:03d}/{args.epochs:03d} "
            f"loss={loss:.4f} train_acc={train_acc:.4f}",
            flush=True,
        )
        history.append(row)

    final_metrics = evaluate(model, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{spec.name.lower()}_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "dataset": spec.name,
            "epoch": args.epochs,
            "metrics": final_metrics,
            "input_bands": input_bands,
        },
        checkpoint_path,
    )
    print(
        f"[{spec.name}] final test "
        f"OA={final_metrics['OA']:.4f} AA={final_metrics['AA']:.4f} "
        f"Kappa={final_metrics['Kappa']:.4f} F1={final_metrics['F1']:.4f}",
        flush=True,
    )

    selected_bands_by_frequency = summarize_selected_bands(model, train_loader, device)
    result = {
        "dataset": spec.name,
        "image_shape": list(t1.shape),
        "train_samples": int(len(train_set)),
        "test_samples": int(len(test_set)),
        "input_bands": int(input_bands),
        "keep_bands": int(min(args.keep_bands, input_bands)),
        "model": "fixed_sst_aci",
        "fusion_mode": "aci",
        "band_selection": "graph",
        "fixed_band_ranking": args.fixed_band_ranking,
        "selected_bands": selected_bands.tolist() if selected_bands is not None else [],
        "checkpoint": str(checkpoint_path),
        "final_epoch": int(args.epochs),
        "final_metrics": final_metrics,
        "selected_bands_by_frequency": selected_bands_by_frequency[: min(30, len(selected_bands_by_frequency))],
        "seconds": round(time.time() - t0, 2),
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{spec.name.lower()}_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_summary(output_dir: Path, results: list[dict[str, object]]) -> None:
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "fusion_mode",
                "final_epoch",
                "OA",
                "AA",
                "Kappa",
                "F1",
                "train_samples",
                "test_samples",
                "seconds",
            ]
        )
        for result in results:
            metrics = result["final_metrics"] or {}
            writer.writerow(
                [
                    result["dataset"],
                    result.get("fusion_mode", ""),
                    result["final_epoch"],
                    metrics.get("OA", ""),
                    metrics.get("AA", ""),
                    metrics.get("Kappa", ""),
                    metrics.get("F1", ""),
                    result["train_samples"],
                    result["test_samples"],
                    result["seconds"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("B3STFormer for hyperspectral image change detection")
    parser.add_argument("--dataset", default="all", help="China, Barbara, River, BayArea, Farmland, or all")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\wyh\wyh\change_detection_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-number", type=int, default=500, help="samples per class for training")
    parser.add_argument("--max-test-points", type=int, default=20000, help="balanced evaluation cap; <=0 uses all labeled pixels")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--keep-bands", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sst-dim", type=int, default=32)
    parser.add_argument("--sst-depth", type=int, default=2)
    parser.add_argument("--sst-heads", type=int, default=4)
    parser.add_argument("--sst-dim-head", type=int, default=16)
    parser.add_argument("--sst-mlp-dim", type=int, default=8)
    parser.add_argument("--b-dim", type=int, default=256)
    parser.add_argument("--b-depth", type=int, default=2)
    parser.add_argument("--b-heads", type=int, default=4)
    parser.add_argument("--b-dim-head", type=int, default=32)
    parser.add_argument("--b-mlp-dim", type=int, default=64)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--fixed-band-ranking",
        choices=["diverse", "topk"],
        default="diverse",
        help="ranking policy used after graph-guided scoring",
    )
    parser.add_argument("--graph-neighbors", type=int, default=8)
    parser.add_argument("--graph-weight", type=float, default=0.35)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.max_test_points is not None and args.max_test_points <= 0:
        args.max_test_points = None
    args.model = "fixed_sst_aci"
    return args


def main() -> None:
    args = parse_args()
    if args.dataset.lower() == "all":
        datasets = available_datasets()
    else:
        datasets = [DATASET_SPECS_NAME(canonical_dataset_name(args.dataset))]

    args.output_dir = args.output_dir.resolve()
    results = []
    for dataset_name in datasets:
        results.append(run_dataset(args, dataset_name))
    write_summary(args.output_dir, results)
    print(f"Saved results to {args.output_dir}", flush=True)


def DATASET_SPECS_NAME(key: str) -> str:
    from .data import DATASET_SPECS

    return DATASET_SPECS[key].name


if __name__ == "__main__":
    main()
