"""
Configuration file for Dual Stream Pose Estimation Model
"""

class DualStreamConfig:
    """Configuration class for dual stream model"""
    
    def __init__(self):
        # Model Architecture
        self.motion_dim = 384                    # ActionFormer output dimension
        self.joint_feature_dim = 128            # Joint feature dimension
        self.num_joints = 15                    # Number of skeleton joints
        self.num_heads = 8                      # Multi-head attention heads
        self.num_transformer_layers = 3         # Transformer layers for joint processing
        self.heatmap_size = 128                 # Input heatmap size
        self.output_pose_dim = 3                # Output pose dimension (x, y, z)
        
        # Heatmap to Feature Conversion
        self.heatmap_conversion_method = 'conv_pool'  # 'adaptive_pool', 'conv_pool', 'spatial_attention'
        
        # Joint Aggregation
        self.joint_aggregation_method = 'attention'  # 'mean', 'attention', 'flatten'
        self.joint_aggregation_output_dim = 128
        
        # Cross Attention
        self.cross_attention_dropout = 0.1
        
        # Training Parameters
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.gradient_clip_value = 1.0
        self.num_epochs = 50
        self.batch_size = 8
        
        # Scheduler
        self.scheduler_type = 'cosine'          # 'cosine', 'step', 'exponential'
        self.scheduler_eta_min = 5e-5
        
        # Data Parameters
        self.sequence_length = 64
        self.crop_size = 224
        
        # Loss Weights (should match your base options)
        self.lambda_mpjpe = 1.0
        self.lambda_cos_sim = 0.1
        self.lambda_bone_length = 0.1
        
        # Logging and Saving
        self.log_step = 20
        self.save_freq = 5                      # Save model every N epochs
        self.vis_freq = 5                       # Visualize attention every N epochs
        
        # Paths
        self.actionformer_config_path = 'actionformer/config/ego4D_egovlp.yaml'
        
        # Joint Names (customize based on your skeleton)
        self.joint_names = [
            'Head', 'Neck', 'LShoulder', 'LElbow', 'LWrist',
            'RShoulder', 'RElbow', 'RWrist', 'Torso', 'LHip',
            'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle'
        ]
        
        # Bone Connections for loss computation (pairs of joint indices)
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # Head to left arm
            (1, 5), (5, 6), (6, 7),             # Neck to right arm  
            (1, 8), (8, 9), (9, 10),            # Neck to left leg
            (1, 11), (11, 12), (12, 13)         # Neck to right leg
        ]
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'motion_dim': self.motion_dim,
            'joint_feature_dim': self.joint_feature_dim,
            'num_joints': self.num_joints,
            'num_heads': self.num_heads,
            'num_transformer_layers': self.num_transformer_layers,
            'heatmap_size': self.heatmap_size,
            'output_pose_dim': self.output_pose_dim
        }
    
    def update_from_args(self, args):
        """Update config from command line arguments"""
        if hasattr(args, 'embed_feature_dim'):
            self.motion_dim = args.embed_feature_dim
        if hasattr(args, 'joint_feature_dim'):
            self.joint_feature_dim = args.joint_feature_dim
        if hasattr(args, 'num_heads'):
            self.num_heads = args.num_heads
        if hasattr(args, 'num_transformer_layers'):
            self.num_transformer_layers = args.num_transformer_layers
        if hasattr(args, 'learning_rate'):
            self.learning_rate = args.learning_rate
        if hasattr(args, 'batch_size'):
            self.batch_size = args.batch_size
        if hasattr(args, 'num_epochs'):
            self.num_epochs = args.num_epochs
        if hasattr(args, 'crop_size'):
            self.crop_size = args.crop_size
        if hasattr(args, 'config_path'):
            self.actionformer_config_path = args.config_path


# Different model configurations for experimentation
class LightweightConfig(DualStreamConfig):
    """Lightweight configuration for faster training/inference"""
    
    def __init__(self):
        super().__init__()
        self.joint_feature_dim = 64
        self.num_heads = 4
        self.num_transformer_layers = 2
        self.joint_aggregation_output_dim = 64


class HighCapacityConfig(DualStreamConfig):
    """High capacity configuration for better performance"""
    
    def __init__(self):
        super().__init__()
        self.joint_feature_dim = 256
        self.num_heads = 16
        self.num_transformer_layers = 6
        self.joint_aggregation_output_dim = 256
        self.learning_rate = 5e-5  # Lower LR for larger model


class ExperimentalConfig(DualStreamConfig):
    """Configuration for experimental features"""
    
    def __init__(self):
        super().__init__()
        self.heatmap_conversion_method = 'spatial_attention'
        self.joint_aggregation_method = 'flatten'
        self.cross_attention_dropout = 0.2


def get_config(config_name='default'):
    """
    Get configuration by name
    
    Args:
        config_name: 'default', 'lightweight', 'high_capacity', 'experimental'
    
    Returns:
        config: Configuration object
    """
    if config_name == 'default':
        return DualStreamConfig()
    elif config_name == 'lightweight':
        return LightweightConfig()
    elif config_name == 'high_capacity':
        return HighCapacityConfig()
    elif config_name == 'experimental':
        return ExperimentalConfig()
    else:
        raise ValueError(f"Unknown config name: {config_name}")


if __name__ == "__main__":
    # Test configurations
    configs = ['default', 'lightweight', 'high_capacity', 'experimental']
    
    for config_name in configs:
        print(f"\n=== {config_name.upper()} CONFIG ===")
        config = get_config(config_name)
        print(f"Motion dim: {config.motion_dim}")
        print(f"Joint feature dim: {config.joint_feature_dim}")
        print(f"Num heads: {config.num_heads}")
        print(f"Num transformer layers: {config.num_transformer_layers}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Heatmap conversion: {config.heatmap_conversion_method}")
        print(f"Joint aggregation: {config.joint_aggregation_method}")
