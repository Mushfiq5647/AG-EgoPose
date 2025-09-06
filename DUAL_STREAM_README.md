# Dual Stream Pose Estimation Model with Cross Attention

This repository contains a comprehensive implementation of a dual-stream pose estimation model that combines motion understanding from ActionFormer with spatial joint relationships through cross attention.

## 🏗️ Architecture Overview

The dual stream model consists of:

1. **Motion Stream**: ActionFormer processes temporal image features for motion understanding
2. **Pose Stream**: Spatial transformer processes joint relationships from heatmaps  
3. **Cross Attention**: Allows motion features to selectively attend to relevant joints
4. **Fusion**: Combines both streams for final pose prediction

```
Images → ActionFormer → Motion Features (B, T, 384)
                           ↓
Heatmaps → Joint Transformer → Joint Features (B, T, J, 128)
                           ↓
              Cross Attention Fusion
                           ↓
              Pose Decoder → 3D Poses (B, T, J, 3)
```

## 📁 File Structure

```
├── utils/
│   └── cross_attention_model.py      # Main dual stream model implementation
├── train_dual_stream.py              # Training script
├── evaluate_dual_stream.py           # Evaluation script  
├── dual_stream_config.py             # Configuration management
├── run_dual_stream_example.py        # Usage examples and demos
└── DUAL_STREAM_README.md             # This file
```

## 🔧 Components

### 1. HeatmapToJointFeatures
Converts 2D heatmaps to joint features using different methods:
- `adaptive_pool`: Simple global average pooling
- `conv_pool`: Convolutional feature extraction (recommended)
- `spatial_attention`: Attention-weighted pooling

### 2. SpatialJointTransformer  
Transformer encoder that models relationships between joints:
- Processes joints as sequence elements
- Learns anatomical constraints automatically
- Includes positional embeddings for joint structure

### 3. MotionPoseCrossAttention
Cross attention mechanism allowing motion features to attend to pose features:
- Motion features query joint features
- Learns which joints are important for different motions
- Provides interpretable attention weights

### 4. JointAggregator
Aggregates joint features for final fusion:
- `mean`: Simple mean pooling
- `attention`: Learned attention weights (recommended)
- `flatten`: Concatenate all joint features

### 5. DualStreamPoseEstimator
Complete end-to-end model combining all components.

## 🚀 Quick Start

### 1. Installation
```bash
# Install required packages
pip install torch torchvision matplotlib seaborn
```

### 2. Configuration
Choose a configuration based on your needs:

```python
from dual_stream_config import get_config

# Options: 'default', 'lightweight', 'high_capacity', 'experimental'
config = get_config('default')
```

### 3. Training

#### Basic Training (with pre-trained heatmaps):
```bash
python train_dual_stream.py \
  --model_path ./models/dual_stream_basic \
  --annotation_path ./data/train_annotation.pkl \
  --heatmap_trained_path ./utils/trained_heatmaps/heatmap--006.ckpt \
  --num_epochs 20 \
  --batch_size 16 \
  --learning_rate 1e-4
```

#### Training without pre-trained heatmaps (uses GT):
```bash
python train_dual_stream.py \
  --model_path ./models/dual_stream_gt \
  --annotation_path ./data/train_annotation.pkl \
  --num_epochs 30
```

### 4. Evaluation
```bash
python evaluate_dual_stream.py \
  --model_path ./models/dual_stream_basic \
  --checkpoint_path ./models/dual_stream_basic/dual_stream_epoch_50.pth \
  --annotation_path ./data/test_annotation.pkl \
  --heatmap_trained_path ./models/heatmap_model.pth
```

## ⚙️ Configuration Options

### Model Configurations

| Configuration | Joint Dim | Heads | Layers | Use Case |
|--------------|-----------|-------|--------|----------|
| Lightweight  | 64        | 4     | 2      | Fast experiments |
| Default      | 128       | 8     | 3      | Balanced performance |
| High Capacity| 256       | 16    | 6      | Maximum accuracy |

### Training Parameters

```python
# Key parameters you can adjust:
--joint_feature_dim 128        # Dimension of joint features
--num_heads 8                  # Number of attention heads  
--num_transformer_layers 3     # Transformer layers for joints
--learning_rate 1e-4           # Initial learning rate
--batch_size 8                 # Batch size
--num_epochs 50                # Training epochs
```

## 📊 Evaluation Metrics

The model provides comprehensive evaluation:

- **MPJPE**: Mean Per Joint Position Error
- **PA-MPJPE**: Procrustes-aligned MPJPE  
- **Bone Length Error**: Consistency of skeletal structure
- **Per-Joint Errors**: Error breakdown by joint
- **Attention Analysis**: Which joints the model focuses on

## 🎯 Key Features

### 1. Cross Attention Visualization
The model provides interpretable attention weights showing which joints are important for different motions:

```python
# After evaluation, attention weights show:
# Walking: High attention on hip, knee, ankle joints
# Reaching: High attention on shoulder, elbow, wrist joints  
# Sitting: High attention on hip, torso joints
```

### 2. Flexible Heatmap Input
Can work with either:
- **Predicted heatmaps**: From pre-trained 2D pose estimation
- **Ground truth heatmaps**: For oracle experiments

### 3. Modular Design
Each component can be used independently:
- Replace joint transformer with GNN
- Try different heatmap conversion methods
- Experiment with fusion strategies

### 4. Memory Efficient
- ActionFormer features frozen to save memory
- Gradient checkpointing support
- Efficient attention computation

## 🔬 Experimental Features

### Ablation Studies
Test different components:

```bash
# Test without cross attention (simple concatenation)
--use_cross_attention False

# Test different joint aggregation methods
--joint_aggregation_method attention  # or 'mean', 'flatten'

# Test different heatmap conversion methods  
--heatmap_conversion_method conv_pool  # or 'adaptive_pool', 'spatial_attention'
```

### Advanced Training
```bash
# Curriculum learning: start with GT heatmaps, transition to predicted
--curriculum_learning True
--curriculum_epochs 10

# Mixed precision training
--use_amp True

# Advanced augmentation
--advanced_augmentation True
```

## 📈 Expected Results

Based on typical pose estimation benchmarks:

| Method | MPJPE (mm) | PA-MPJPE (mm) | Notes |
|--------|------------|---------------|--------|
| Baseline (GT heatmaps) | ~45 | ~35 | Upper bound |
| With Predicted Heatmaps | ~55 | ~42 | Realistic performance |
| Cross Attention Boost | ~-5% | ~-8% | Improvement from attention |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Use lightweight config: `--joint_feature_dim 64`
   - Enable gradient checkpointing

2. **Poor Attention Patterns**
   - Check heatmap quality first
   - Increase attention heads: `--num_heads 16`
   - Add attention visualization: `--vis_freq 1`

3. **Slow Training**
   - Use lightweight config for prototyping
   - Reduce sequence length: `--seq_length 32`
   - Use mixed precision: `--use_amp True`

### Debugging Tips

```python
# Enable detailed logging
--log_step 1

# Save attention visualizations frequently  
--vis_freq 1

# Monitor gradient norms
--clip_value 0.5
```

## 🔍 Understanding Attention

The cross attention mechanism learns interpretable patterns:

### Motion-Specific Attention
- **Locomotion**: Focuses on leg joints (hip, knee, ankle)
- **Manipulation**: Focuses on arm joints (shoulder, elbow, wrist)  
- **Postural**: Focuses on core joints (spine, torso)

### Debugging Attention
```python
# Visualize attention patterns
attention_weights = model.get_attention_weights()
plot_attention_heatmap(attention_weights, joint_names)

# Analyze attention statistics
attention_stats = analyze_attention_patterns(attention_weights)
print(f"Most attended joint: {attention_stats['top_joint']}")
```

## 🚀 Advanced Usage

### Custom Joint Configurations
```python
# Modify joint names and connections in config
config.joint_names = ['Head', 'Neck', ...]  
config.bone_connections = [(0, 1), (1, 2), ...]
```

### Integration with Existing Models
```python
from utils.cross_attention_model import create_dual_stream_model

# Use with your own ActionFormer
actionformer = YourActionFormer(...)
model = create_dual_stream_model(actionformer, config)
```

### Custom Loss Functions
```python
# Add custom losses in training script
custom_loss = YourCustomLoss()
total_loss += lambda_custom * custom_loss(predictions, targets)
```

## 📚 References

- ActionFormer: Localizing Moments of Actions with Transformers
- Vision Transformer for pose estimation
- Cross-attention mechanisms in computer vision
- Spatio-temporal transformers for action recognition

## 🤝 Contributing

Feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Share experimental results

## 📄 License

[Your license information here]

---

For more details, run the demo script:
```bash
python run_dual_stream_example.py
```

This will show working examples of all components and provide usage guidance.
