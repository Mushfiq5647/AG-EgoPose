# [You2Me: Inferring Body Pose in Egocentric Video via First and Second Person Interactions (CVPR 2020)](http://vision.cs.utexas.edu/projects/you2me/) 

[![report](https://img.shields.io/badge/arXiv-1904.09882-b31b1b.svg)](https://arxiv.org/abs/1904.09882#)

![](data/you2me_preview.gif)


## Install
Download [dataset](https://github.com/facebookresearch/you2me/tree/master/data)

Original training done with CUDA 10.2 

Install basic dependencies with `pip install -r requirements.txt`


## Test
Please generate:

- directory of homographies (see calc_homgraphy/README.md)
- directory of openpose predictions
- vocab.pkl (see vocab/build_vocab.py) 

for your sample sequence.

Then run the following command:

`python sample.py --encoder_path ./utils/kinect_trained_ckpt_you2megcn/encoder-20-66.ckpt --decoder_path ./utils/kinect_trained_ckpt_you2megcn/decoder-20-66.ckpt --visualize`

Change flag `--upp` to `--low` to test the lower body model.

Include flag `--visualize` to plot the predicted stick figures.

## Train
Please generate 

- directory of homographies (see calc_homgraphy/README.md)
- directory of openpose predictions
- vocab.pkl (see vocab/build_vocab.py)
- annotation.pkl (see vocab/build_annotation.py)

for your each of your training sequences.

Then run the following command:

`python train.py --model_path ./utils/model.py --annotation_path ./vocab/train_annotation.pkl --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt`
`python train_finetune.py --model_path ./utils/model.py --annotation_path ./vocab/train_annotation.pkl --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt`
For testing:
`python test.py --encoder_path ./utils/mo2cap2/encoder-finetune-20-848.ckpt --decoder_path ./utils/mo2cap2/pose-decoder-finetune-20-848.ckpt --heatmap_trained_path utils/trained_heatmaps/mse/heatmap_epoch_25.ckpt --heatmap_path ./utils/mo2cap2/heatmap_embedding-finetune-20-848.ckpt --spatial_transformer_path ./utils/mo2cap2/spatial_transformer-finetune-20-848.ckpt`
`python test.py --encoder_path ./utils/sceneego/encoder-best.ckpt --decoder_path ./utils/sceneego/pose-decoder-best.ckpt --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt --heatmap_path ./utils/sceneego/heatmap_embedding-best.ckpt --spatial_transformer_path ./utils/sceneego/spatial_transformer-best.ckpt`
`python test.py --encoder_path ./utils/trained_egopwtrain_egogta/encoder-best.ckpt --decoder_path ./utils/trained_egopwtrain_egogta/pose-decoder-best.ckpt --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt --heatmap_path ./utils/trained_egopwtrain_bce_motion/heatmap_embedding-best.ckpt --spatial_transformer_path ./utils/trained_egopwtrain_bce_motion/spatial_transformer-best.ckpt`
`python test.py --encoder_path ./utils/trained_unrealego_bce/encoder-best.ckpt --decoder_path ./utils/trained_unrealego_bce/pose-decoder-best.ckpt --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt --heatmap_path ./utils/trained_unrealego_bce/heatmap_embedding-best.ckpt --spatial_transformer_path ./utils/trained_unrealego_bce/spatial_transformer-best.ckpt`
`python visualize_3d_poses.py --encoder_path ./utils/trained_egopwtrain_bce/encoder-best.ckpt --decoder_path ./utils/trained_egopwtrain_bce/pose-decoder-best.ckpt --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt --heatmap_path ./utils/trained_egopwtrain_bce/heatmap_embedding-best.ckpt --spatial_transformer_path ./utils/trained_egopwtrain_bce/spatial_transformer-best.ckpt`
`python quick_pose_visualization.py --encoder_path ./utils/sceneego/encoder-best.ckpt --decoder_path ./utils/sceneego/pose-decoder-best.ckpt --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt --heatmap_path ./utils/sceneego/heatmap_embedding-best.ckpt --spatial_transformer_path ./utils/sceneego/spatial_transformer-best.ckpt`
`python visualize_single_pose_egopw.py --encoder_path ./utils/trained_egopwtrain_bce/encoder-best.ckpt --decoder_path ./utils/trained_egopwtrain_bce/pose-decoder-best.ckpt --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt --heatmap_path ./utils/trained_egopwtrain_bce/heatmap_embedding-best.ckpt --spatial_transformer_path ./utils/trained_egopwtrain_bce/spatial_transformer-best.ckpt`
`python visualize_single_pose.py --encoder_path ./utils/sceneego/encoder-best.ckpt --decoder_path ./utils/sceneego/pose-decoder-best.ckpt --heatmap_trained_path utils/trained_heatmaps/bce_combined/heatmap_best.ckpt --heatmap_path ./utils/sceneego/heatmap_embedding-best.ckpt --spatial_transformer_path ./utils/sceneego/spatial_transformer-best.ckpt`
Change flag `--upp` to `--low` to train the lower body model.

## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 


## Citation

```
@article{ng2019you2me,
  title={You2Me: Inferring Body Pose in Egocentric Video via First and Second Person Interactions},
  author={Ng, Evonne and Xiang, Donglai and Joo, Hanbyul and Grauman, Kristen},
  journal={CVPR},
  year={2020}
}
```