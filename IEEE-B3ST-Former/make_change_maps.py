from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from hsi_cd.data import DATASET_SPECS, HSIPatchPairDataset, canonical_dataset_name, load_dataset_pair
from hsi_cd.metrics import binary_metrics
from hsi_cd.models import FixedBandSSTACIDetector, HSIChangeDetector, SSTFormerReference, SelectedSSTACIDetector


def all_pixel_dataset(t1: np.ndarray, t2: np.ndarray, patch_size: int) -> HSIPatchPairDataset:
    h, w, _ = t1.shape
    rows, cols = np.indices((h, w))
    positions = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)
    labels = np.zeros(h * w, dtype=np.int64)
    return HSIPatchPairDataset(t1, t2, positions, labels, patch_size)


def build_model(checkpoint: dict, input_bands: int, patch_size: int) -> torch.nn.Module:
    args = checkpoint["args"]
    state = checkpoint["model"]
    model_name = args.get("model", "fixed_sst_aci")
    if model_name == "fixed_sst_aci":
        selected_bands = state["selected_bands"].detach().cpu()
        model = FixedBandSSTACIDetector(
            input_bands=input_bands,
            patch_size=patch_size,
            selected_bands=selected_bands,
            dim=args.get("sst_dim", 32),
            depth=args.get("sst_depth", 2),
            heads=args.get("sst_heads", 4),
            dim_head=args.get("sst_dim_head", 16),
            mlp_dim=args.get("sst_mlp_dim", 8),
            b_dim=args.get("b_dim", 256),
            b_depth=args.get("b_depth", 2),
            b_heads=args.get("b_heads", 4),
            b_dim_head=args.get("b_dim_head", 32),
            b_mlp_dim=args.get("b_mlp_dim", 64),
            dropout=args.get("dropout", 0.1),
            emb_dropout=args.get("emb_dropout", 0.1),
        )
    elif model_name == "selected_sst_aci":
        model = SelectedSSTACIDetector(
            input_bands=input_bands,
            patch_size=patch_size,
            keep_bands=args.get("keep_bands", 32),
            dim=args.get("sst_dim", 32),
            depth=args.get("sst_depth", 2),
            heads=args.get("sst_heads", 4),
            dim_head=args.get("sst_dim_head", 16),
            mlp_dim=args.get("sst_mlp_dim", 8),
            b_dim=args.get("b_dim", 256),
            b_depth=args.get("b_depth", 2),
            b_heads=args.get("b_heads", 4),
            b_dim_head=args.get("b_dim_head", 32),
            b_mlp_dim=args.get("b_mlp_dim", 64),
            dropout=args.get("dropout", 0.1),
            emb_dropout=args.get("emb_dropout", 0.1),
            selector_rank=args.get("selector_rank", 16),
        )
    elif model_name == "compact":
        model = HSIChangeDetector(
            input_bands=input_bands,
            patch_size=patch_size,
            keep_bands=args.get("keep_bands", 32),
            dim=args.get("dim", 48),
            depth=args.get("depth", 1),
            heads=args.get("heads", 4),
            mlp_dim=args.get("mlp_dim", 96),
            dropout=args.get("dropout", 0.1),
            selector_rank=args.get("selector_rank", 16),
        )
    elif model_name == "sstformer":
        model = SSTFormerReference(
            input_bands=input_bands,
            patch_size=patch_size,
            dim=args.get("sst_dim", 32),
            depth=args.get("sst_depth", 2),
            heads=args.get("sst_heads", 4),
            dim_head=args.get("sst_dim_head", 16),
            mlp_dim=args.get("sst_mlp_dim", 8),
            b_dim=args.get("b_dim", 512),
            b_depth=args.get("b_depth", 3),
            b_heads=args.get("b_heads", 8),
            b_dim_head=args.get("b_dim_head", 32),
            b_mlp_dim=args.get("b_mlp_dim", 8),
            cross_depth=args.get("cross_depth", 3),
            dropout=args.get("dropout", 0.2),
            emb_dropout=args.get("emb_dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model in checkpoint: {model_name}")
    model.load_state_dict(state)
    return model


@torch.no_grad()
def predict_full_map(
    model: torch.nn.Module,
    dataset: HSIPatchPairDataset,
    image_shape: tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model.eval()
    preds: list[np.ndarray] = []
    for x1, x2, _ in loader:
        logits = model(x1.to(device, non_blocking=True), x2.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8))
    return np.concatenate(preds).reshape(image_shape)


def label_to_binary(label: np.ndarray, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = DATASET_SPECS[canonical_dataset_name(dataset_name)]
    valid = (label == spec.changed_value) | (label == spec.unchanged_value)
    binary = np.zeros(label.shape, dtype=np.uint8)
    binary[label == spec.changed_value] = 1
    return binary, valid


def save_binary(path: Path, arr: np.ndarray) -> None:
    Image.fromarray((arr.astype(np.uint8) * 255)).save(path)


def error_rgb(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb = np.full((*pred.shape, 3), 128, dtype=np.uint8)
    tn = valid & (gt == 0) & (pred == 0)
    tp = valid & (gt == 1) & (pred == 1)
    fp = valid & (gt == 0) & (pred == 1)
    fn = valid & (gt == 1) & (pred == 0)
    rgb[tn] = np.array([0, 0, 0], dtype=np.uint8)
    rgb[tp] = np.array([255, 255, 255], dtype=np.uint8)
    rgb[fp] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[fn] = np.array([0, 80, 255], dtype=np.uint8)
    return rgb


def labeled_panel(title: str, image: Image.Image, width: int) -> Image.Image:
    header_h = 30
    panel = Image.new("RGB", (width, image.height + header_h), "white")
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(20, 20, 20), font=font)
    panel.paste(image.convert("RGB"), (0, header_h))
    return panel


def save_composite(path: Path, gt: np.ndarray, pred: np.ndarray, err: np.ndarray) -> None:
    scale_w = gt.shape[1]
    gt_img = Image.fromarray(gt.astype(np.uint8) * 255).convert("RGB")
    pred_img = Image.fromarray(pred.astype(np.uint8) * 255).convert("RGB")
    err_img = Image.fromarray(err).convert("RGB")
    panels = [
        labeled_panel("Ground Truth", gt_img, scale_w),
        labeled_panel("Prediction", pred_img, scale_w),
        labeled_panel("Error Map  red=FP, blue=FN", err_img, scale_w),
    ]
    composite = Image.new("RGB", (scale_w * 3, panels[0].height), "white")
    for idx, panel in enumerate(panels):
        composite.paste(panel, (idx * scale_w, 0))
    composite.save(path)


def run_one(args: argparse.Namespace, dataset_name: str) -> dict[str, object]:
    dataset_key = canonical_dataset_name(dataset_name)
    spec = DATASET_SPECS[dataset_key]
    checkpoint_path = args.checkpoint_dir / f"{spec.name.lower()}_last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    t1, t2, label, _ = load_dataset_pair(args.data_root, spec.name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    patch_size = int(checkpoint["args"].get("patch_size", args.patch_size))
    model = build_model(checkpoint, input_bands=t1.shape[2], patch_size=patch_size).to(args.device)

    full_dataset = all_pixel_dataset(t1, t2, patch_size)
    pred = predict_full_map(
        model=model,
        dataset=full_dataset,
        image_shape=label.shape,
        batch_size=args.batch_size,
        device=args.device,
    )
    gt, valid = label_to_binary(label, spec.name)
    metrics = binary_metrics(gt[valid], pred[valid])

    out_dir = args.output_dir / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_binary(out_dir / "ground_truth.png", gt)
    save_binary(out_dir / "prediction.png", pred)
    err = error_rgb(pred, gt, valid)
    Image.fromarray(err).save(out_dir / "error_map.png")
    save_composite(out_dir / "visual_comparison.png", gt, pred, err)
    np.save(out_dir / "prediction.npy", pred)

    result = {
        "dataset": spec.name,
        "checkpoint": str(checkpoint_path),
        "image_shape": list(label.shape),
        "metrics_full_valid_pixels": metrics,
        "paths": {
            "ground_truth": str(out_dir / "ground_truth.png"),
            "prediction": str(out_dir / "prediction.png"),
            "error_map": str(out_dir / "error_map.png"),
            "visual_comparison": str(out_dir / "visual_comparison.png"),
            "prediction_npy": str(out_dir / "prediction.npy"),
        },
    }
    with (out_dir / "map_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate full-image prediction and error maps")
    parser.add_argument("--datasets", nargs="+", default=["China", "Barbara", "River"])
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\wyh\wyh\change_detection_data"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_one(args, dataset_name) for dataset_name in args.datasets]
    with (args.output_dir / "map_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
