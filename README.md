# B3STFormer

Official code release for **B3STFormer: Band-Selected Spectral-Spatial Transformer with Temporal Adaptive Modulation for Hyperspectral Image Change Detection**.

B3STFormer contains three main components:

- **SDBS**: spectral-aware discriminative band selection for selecting a unified band subset from two temporal HSIs.
- **Spectral-spatial Transformer**: shared encoder for extracting spectral and spatial features from bi-temporal patch pairs.
- **TAM**: temporal adaptive modulation for change-sensitive bi-temporal feature fusion.

This cleaned release keeps the main training, testing, and change-map generation workflow.

## Requirements

```bash
pip install -r requirements.txt
```

The experiments were developed with Python 3.10 and PyTorch. A CUDA-enabled GPU is recommended.

## Dataset Layout

Set `--data-root` to the directory containing the HSI-CD datasets. The expected layout is:

```text
change_detection_data/
  China_change_detection/
    China_Change_Dataset.mat
  Barbara/
    barbara_2013.mat
    barbara_2014.mat
    barbara_gtChanges.mat
  river_dataset/
    river_before.mat
    river_after.mat
    groundtruth.mat
```

The loader also contains local support for BayArea and Farmland.

## Training and Testing

The training script trains the model for the specified number of epochs, saves the last-epoch checkpoint, and evaluates the test set once after training.

Main paper setting: patch size `P=5`, selected bands `K=64`, 500 changed and 500 unchanged training samples, SDBS band selection, and TAM fusion.

```bash
python run_experiments.py \
  --dataset China \
  --data-root /path/to/change_detection_data \
  --output-dir outputs/b3stformer \
  --epochs 200 \
  --train-number 500 \
  --max-test-points 0 \
  --patch-size 5 \
  --keep-bands 64 \
  --fixed-band-ranking diverse \
  --graph-neighbors 8 \
  --graph-weight 0.35
```

Run the same command with `--dataset Barbara` or `--dataset River` for the other datasets.

The output directory contains:

```text
china_last.pt
china_result.json
summary.csv
```

## Change Maps

After training the datasets into the same checkpoint directory, generate prediction and error maps with:

```bash
python make_change_maps.py \
  --datasets China Barbara River \
  --data-root /path/to/change_detection_data \
  --checkpoint-dir outputs/b3stformer \
  --output-dir outputs/change_maps
```

For each dataset, the script saves:

```text
ground_truth.png
prediction.png
error_map.png
visual_comparison.png
prediction.npy
map_metrics.json
```

The error map uses black for TN, white for TP, red for FP, and blue for FN.

## Notes

- Raw datasets and trained checkpoints are not included in this repository.
- Please add a final license file before making the repository public.
