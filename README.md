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

## Pretrained Models

### ActionFormer weights

The motion encoder loads an Ego4D-pretrained ActionFormer checkpoint. Download it from the following link and place it at:

- `actionformer/ego4d_egovlp_reproduce/epoch_010.pth.tar`

[Download ActionFormer checkpoint](https://drive.google.com/drive/folders/1NpAECS0ZhcCuehXkF9OhLQDPFrNdStJb) 

### Pre-trained Models

If you don't want to train from scratch, you can download the following pre-trained models:

- **2D Heatmap Network**: [Download](https://drive.google.com/drive/u/0/folders/1nmdQzYoh_oe4MZxyx-rBDbcx69osVUTf)
  - Place at: `utils/trained_heatmaps/bce_combined/heatmap_best.ckpt`

- **AG-EgoPose Model**: [Download](https://drive.google.com/drive/u/0/folders/1p6uUM3bW4yC-dzV92mFS3g03zXoTv13r) 
  - Place the checkpoint files in `utils/trained_model/` directory with the following structure:
    - `utils/trained_model/encoder-best.ckpt`
    - `utils/trained_model/pose-decoder-best.ckpt`
    - `utils/trained_model/heatmap_embedding-best.ckpt`

## Datasets

### Download Datasets

Download the EgoPW and SceneEgo datasets from the following links:

- **EgoPW Dataset**: [Download](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.FQAEOV) 
- **SceneEgo Dataset**: [Download](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.VCIHDO) 

### Data setup

The dataloader reads **sequence root directories** from text files named:

- `train_egopw.txt`, `test_egopw.txt`
- `train_sceneego.txt`, `test_sceneego.txt`

Example list files are provided in `train_test_directory/`.

## Expected dataset layout

After downloading the datasets, each sequence directory should have the following structure:

### EgoPW
Each directory listed in `train_egopw.txt` / `test_egopw.txt` should contain:

- `imgs/` (RGB frames named `img_XXXXXX.jpg`)
- `pseudo_gt.pkl` (or `pseudo.pkl`) for GT pose lookup

### SceneEgo
Each directory listed in `train_sceneego.txt` / `test_sceneego.txt` should contain:

- `imgs/` (RGB frames)
- `local_pose_gt.pkl`
- `syn.json`

**Note**: The heatmaps are not included in the downloaded datasets. You must generate them using the scripts in the "Generate heatmaps" section below, which will create the `heatmap/` folder automatically.

## Generate heatmaps (EgoPW / SceneEgo)

**Important**: Before training or evaluation, you must generate 2D heatmaps for your datasets. The heatmap generation scripts will automatically create a `heatmap/` folder inside each sequence directory containing the heatmap `.npy` files.

### EgoPW

Generate heatmaps for EgoPW dataset. The script creates `heatmap/` folder by default (64×64, sigma=2.0):

```bash
python heatmaps/generate_heatmaps/generate_heatmaps_egopw.py \
  --train_txt train_egopw.txt \
  --hm_size 64 \
  --sigma 2.0
```

The script will process each sequence directory listed in `train_egopw.txt` and create `heatmap/heatmap_XXXXXX.npy` files inside each sequence folder.

### SceneEgo

Generate heatmaps for SceneEgo dataset from 3D pose via fisheye projection. The script creates `heatmap/` folder:

```bash
python heatmaps/generate_heatmaps/generate_heatmaps_sceneego.py \
  --train_txt train_sceneego.txt \
  --fisheye_json 2D_projection/fisheye.json \
  --hm_size 64 \
  --sigma 2.0
```

The script will process each sequence directory listed in `train_sceneego.txt` and create `heatmap/heatmap_XXXXXX.npy` files inside each sequence folder.

## Train

### 1) (Optional) Train the 2D heatmap network

If you downloaded the pre-trained 2D heatmap model (see "Pretrained Models" section above) or already have a trained checkpoint at `utils/trained_heatmaps/bce_combined/heatmap_best.ckpt`, you can skip this step.

To train from scratch:

```bash
python heatmaps/train_2D_heatmaps.py
```

By default, this writes:
- `utils/trained_heatmaps/bce_combined/heatmap_best.ckpt`

Note: this script uses the repo's `TrainOptions`/dataloader and assumes your dataset list files and heatmap folders are set up as described above.

### 2) Train AG-EgoPose

```bash
python train.py \
  --model_path utils/trained_model \
  --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt \
  --config_path actionformer/config/ego4D_egovlp.yaml
```

Checkpoints are saved under `--model_path`:
- `encoder-best.ckpt`
- `pose-decoder-best.ckpt`
- `heatmap_embedding-best.ckpt`


### 3) Fine-tune (optional)

```bash
python train_finetune.py \
  --model_path utils/exp_finetune \
  --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt \
  --encoder_path utils/trained_model/encoder-best.ckpt \
  --decoder_path utils/trained_model/pose-decoder-best.ckpt \
  --heatmap_path utils/trained_model/heatmap_embedding-best.ckpt \
```

## Evaluate

```bash
python test.py \
  --encoder_path utils/trained_model/encoder-best.ckpt \
  --decoder_path utils/trained_model/pose-decoder-best.ckpt \
  --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt \
  --heatmap_path utils/trained_model/heatmap_embedding-best.ckpt \
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Md Mushfiqur Azam


