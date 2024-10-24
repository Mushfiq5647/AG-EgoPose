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

`python sample.py --vocab_path ./vocab/vocab_py3.pkl --output ./outputs --encoder_path ./utils/trained_ckpt_actionformer/encoder-20-61.ckpt --decoder_path ./utils/trained_ckpt_actionformer/decoder-20-61.ckpt --upp --visualize`

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

`python train.py --model_path ./utils/model.py --annotation_path ./vocab/train_annotation.pkl --validation_annotation_path ./vocab/validation_annotation.pkl`


For testing:
`python test.py --encoder_path ./utils/trained_ckpt_actionformer/encoder-20-61.ckpt --decoder_path ./utils/trained_ckpt_actionformer/decoder-20-61.ckpt --output_dir ./outputs --vocab_path ./vocab/vocab_py3.pkl --test_annotation_path ./vocab/test_annotation.pkl`

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