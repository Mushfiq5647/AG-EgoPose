## AG-EgoPose

AG-EgoPose is a dual-stream framework for **egocentric 3D human pose estimation** from monocular fisheye video.

- **Spatial stream**: ResNet-18 encoder-decoder predicts **2D joint heatmaps** (15×64×64).
- **Temporal stream**: ResNet-50 frame features are processed by **ActionFormer** (Ego4D-pretrained, used as a temporal encoder) to obtain motion features.
- **Pose regression**: Heatmap-derived joint tokens + motion features are fused to predict **3D joints**.

This repository contains training (`train.py`, `train_finetune.py`), evaluation (`test.py`), analysis (`model_analysis/`), and visualization (`visualization/`) utilities.

## Installation

```bash
pip install -r requirements.txt
```

You may also need (depending on scripts): `opencv-python`, `tqdm`, `natsort`, `Pillow`, `h5py`.

## ActionFormer weights

The motion encoder loads an Ego4D-pretrained ActionFormer checkpoint. Make sure this exists:

- `actionformer/ego4d_egovlp_reproduce/epoch_010.pth.tar`

## Data setup

The dataloader reads **sequence root directories** from text files named:

- `train_egopw.txt`, `test_egopw.txt`
- `train_sceneego.txt`, `test_sceneego.txt`

Example list files are provided in `train_test_directory/`. The loader only searches the **current working directory** or the repo root, so do one of:

### Option A (recommended): symlink the list files into repo root

```bash
ln -sf train_test_directory/train_egopw.txt train_egopw.txt
ln -sf train_test_directory/test_egopw.txt  test_egopw.txt
ln -sf train_test_directory/train_sceneego.txt train_sceneego.txt
ln -sf train_test_directory/test_sceneego.txt  test_sceneego.txt
```

### Option B: run scripts from `train_test_directory/`

```bash
cd train_test_directory
python ../train.py ...
```

## Expected dataset layout

### EgoPW
Each directory listed in `train_egopw.txt` / `test_egopw.txt` should contain:

- `imgs/` (RGB frames named `img_XXXXXX.jpg`)
- `pseudo_gt.pkl` (or `pseudo.pkl`) for GT pose lookup
- `heatmap64_2.0/` containing `heatmap_XXXXXX.npy` (shape 15×64×64) used by the dataloader

### SceneEgo
Each directory listed in `train_sceneego.txt` / `test_sceneego.txt` should contain:

- `imgs/`
- `local_pose_gt.pkl`
- `syn.json`
- `heatmap64_2.0/` containing `heatmap_XXXXXX.npy`

If your precomputed heatmaps are stored in a different folder name (e.g., `heatmap64_1.2/`), rename/symlink it to `heatmap64_2.0/` or update the path in `utils/data_loader.py`.

## Generate heatmaps (EgoPW / SceneEgo)

Most training/evaluation code paths expect precomputed 2D heatmaps on disk (`heatmap64_2.0/heatmap_XXXXXX.npy`).

### EgoPW

Generates `heatmap64_2.0/` by default (64×64, sigma=2.0):

```bash
python heatmaps/generate_heatmaps/generate_heatmaps_egopw.py \
  --train_txt train_egopw.txt \
  --hm_size 64 \
  --sigma 2.0
```

If you use a different sigma, the script can create a different folder name, e.g.:

```bash
python heatmaps/generate_heatmaps/generate_heatmaps_egopw.py \
  --train_txt train_egopw.txt \
  --hm_size 64 \
  --sigma 1.2 \
  --out_dirname heatmap64_2.0
```

### SceneEgo

Generates heatmaps from 3D pose via fisheye projection:

```bash
python heatmaps/generate_heatmaps/generate_heatmaps_sceneego.py \
  --train_txt train_sceneego.txt \
  --fisheye_json 2D_projection/fisheye.json \
  --hm_size 64 \
  --sigma 2.0
```

## Train

### 1) (Optional) Train the 2D heatmap network

If you already have a trained checkpoint (e.g., `utils/trained_heatmaps/bce_combined/heatmap_best.ckpt`), you can skip this step.

```bash
python heatmaps/train_2D_heatmaps_simple.py
```

By default, this writes:
- `utils/trained_heatmaps/bce_combined/heatmap_best.ckpt`

Note: this script uses the repo's `TrainOptions`/dataloader and assumes your dataset list files and heatmap folders are set up as described above.

### 2) Train AG-EgoPose

```bash
python train.py \
  --model_path utils/exp_egopw_seq32 \
  --annotation_path dummy.pkl \
  --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt \
  --config_path actionformer/config/ego4D_egovlp.yaml
```

Checkpoints are saved under `--model_path`:
- `encoder-best.ckpt`
- `pose-decoder-best.ckpt`
- `heatmap_embedding-best.ckpt`
- `spatial_transformer-best.ckpt`

### 3) Fine-tune (optional)

```bash
python train_finetune.py \
  --model_path utils/exp_finetune \
  --annotation_path dummy.pkl \
  --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt \
  --encoder_path utils/exp_egopw_seq32/encoder-best.ckpt \
  --decoder_path utils/exp_egopw_seq32/pose-decoder-best.ckpt \
  --heatmap_path utils/exp_egopw_seq32/heatmap_embedding-best.ckpt \
  --spatial_transformer_path utils/exp_egopw_seq32/spatial_transformer-best.ckpt
```

## Evaluate

```bash
python test.py \
  --encoder_path utils/exp_egopw_seq32/encoder-best.ckpt \
  --decoder_path utils/exp_egopw_seq32/pose-decoder-best.ckpt \
  --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt \
  --heatmap_path utils/exp_egopw_seq32/heatmap_embedding-best.ckpt \
  --spatial_transformer_path utils/exp_egopw_seq32/spatial_transformer-best.ckpt
```

The script writes summary metrics to `test_results.txt`.

## Visualization

- **2D heatmaps (single image visualization)**:

```bash
python visualization/visualize_heatmaps.py \
  --image /path/to/img.jpg \
  --checkpoint utils/trained_heatmaps/bce_combined/heatmap_best.ckpt
```

- **Single-frame 3D pose visualization**:

```bash
python visualization/visualize_single_pose.py \
  --encoder_path ... \
  --decoder_path ... \
  --heatmap_trained_path ... \
  --heatmap_path ... \
  --spatial_transformer_path ...
```

## Profiling / analysis

- **Latency benchmark**:

```bash
python model_analysis/benchmark_inference.py --encoder_path ... --decoder_path ... --heatmap_trained_path ... --heatmap_path ... --spatial_transformer_path ...
```

- **FLOPs / efficiency utilities**:

```bash
python model_analysis/simple_model_analysis.py
python model_analysis/simple_heatmap_flops.py
```

## License

See `LICENSE`.


