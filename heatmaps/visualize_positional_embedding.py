import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleKinematicEncoding(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
        self.joint_embedding = nn.Embedding(15, 64)  # Each joint gets 64D embedding
        self.parent_embedding = nn.Embedding(15, 64)  # Each parent gets 64D embedding
        self.fc = nn.Linear(128, feature_dim)  # Combine joint + parent embeddings
        
    def forward(self, joint_features):
        B, J, F = joint_features.shape  # (B, 15, feature_dim)
        device = joint_features.device
        
        joint_ids = torch.arange(J, device=device)  # [0, 1, 2, ..., 14]
        parent_ids = torch.tensor(self.kinematic_parents, device=device)
        
        # Get embeddings
        joint_emb = self.joint_embedding(joint_ids)  # (15, 64)
        parent_emb = self.parent_embedding(parent_ids)  # (15, 64)
        
        # Combine joint and parent embeddings
        combined_emb = torch.cat([joint_emb, parent_emb], dim=-1)  # (15, 128)
        pos_encoding = self.fc(combined_emb)  # (15, feature_dim)
        
        # Add to joint features
        return joint_features + pos_encoding.unsqueeze(0)  # (B, 15, feature_dim)

class HierarchicalPositionalEncoding(nn.Module):
    def __init__(self, feature_dim=128, max_depth=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_depth = max_depth
        
        # Joint hierarchy levels (0=root, 1=level1, 2=level2, etc.)
        self.joint_levels = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4]
        
        # Embeddings for each level
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(max_depth + 1, feature_dim // 4) for _ in range(4)
        ])
        
        self.fc = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, joint_features):
        B, J, F = joint_features.shape
        device = joint_features.device
        
        # Get level for each joint
        joint_levels = torch.tensor(self.joint_levels, device=device)
        
        # Create level embeddings
        level_embs = []
        for i, emb_layer in enumerate(self.level_embeddings):
            level_embs.append(emb_layer(joint_levels))
        
        # Combine all level embeddings
        pos_encoding = torch.cat(level_embs, dim=-1)  # (15, feature_dim)
        pos_encoding = self.fc(pos_encoding)
        
        return joint_features + pos_encoding.unsqueeze(0)

def visualize_kinematic_embedding():
    """Visualize the kinematic parent-child relationships"""
    joint_names = [
        "Neck", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder",
        "L_Elbow", "L_Wrist", "R_Hip", "R_Knee", "R_Ankle", "R_Foot",
        "L_Hip", "L_Knee", "L_Ankle", "L_Foot"
    ]
    
    kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Parent-child relationships
    ax1.set_title('Joint Hierarchy - Parent-Child Relationships', fontsize=14, fontweight='bold')
    
    # Create tree-like visualization
    levels = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4]
    x_positions = [0, -3, -2, -1, 3, 2, 1, -3, -2, -1, 0, 3, 2, 1, 0]
    
    colors = plt.cm.Set3(np.linspace(0, 1, 15))
    
    for i, (joint, level, x_pos, parent) in enumerate(zip(joint_names, levels, x_positions, kinematic_parents)):
        # Plot joint
        ax1.scatter(x_pos, -level, c=[colors[i]], s=200, alpha=0.8, edgecolors='black', linewidth=2)
        ax1.annotate(f'{i}: {joint}', (x_pos, -level), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
        
        # Draw connection to parent
        if parent != i:  # Not self-connection
            parent_level = levels[parent]
            parent_x = x_positions[parent]
            ax1.plot([parent_x, x_pos], [-parent_level, -level], 'k-', alpha=0.6, linewidth=1.5)
    
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-5, 1)
    ax1.set_xlabel('Joint Position')
    ax1.set_ylabel('Hierarchy Level')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parent mapping matrix
    ax2.set_title('Parent-Child Mapping Matrix', fontsize=14, fontweight='bold')
    
    # Create adjacency matrix
    adj_matrix = np.zeros((15, 15))
    for i, parent in enumerate(kinematic_parents):
        if parent != i:  # Not self-connection
            adj_matrix[i, parent] = 1
    
    im = ax2.imshow(adj_matrix, cmap='Blues', aspect='auto')
    
    # Add labels
    ax2.set_xticks(range(15))
    ax2.set_yticks(range(15))
    ax2.set_xticklabels([f'{i}' for i in range(15)], fontsize=8)
    ax2.set_yticklabels([f'{i}' for i in range(15)], fontsize=8)
    ax2.set_xlabel('Parent Joint Index')
    ax2.set_ylabel('Child Joint Index')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('kinematic_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_embedding_heatmaps():
    """Visualize the learned embeddings"""
    device = torch.device('cpu')
    
    # Initialize models
    kinematic_enc = SimpleKinematicEncoding(feature_dim=128)
    hierarchical_enc = HierarchicalPositionalEncoding(feature_dim=128)
    
    # Create dummy input
    B, J, F = 1, 15, 128
    joint_features = torch.randn(B, J, F)
    
    # Get embeddings
    with torch.no_grad():
        kinematic_pos = kinematic_enc(joint_features) - joint_features  # Just the positional part
        hierarchical_pos = hierarchical_enc(joint_features) - joint_features
    
    joint_names = [
        "Neck", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder",
        "L_Elbow", "L_Wrist", "R_Hip", "R_Knee", "R_Ankle", "R_Foot",
        "L_Hip", "L_Knee", "L_Ankle", "L_Foot"
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Kinematic embedding heatmap
    im1 = ax1.imshow(kinematic_pos[0].numpy(), cmap='RdBu_r', aspect='auto')
    ax1.set_title('SimpleKinematicEncoding - Positional Embeddings', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(15))
    ax1.set_yticklabels([f'{i}: {name}' for i, name in enumerate(joint_names)], fontsize=8)
    ax1.set_xlabel('Feature Dimension')
    ax1.set_ylabel('Joint')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot 2: Hierarchical embedding heatmap
    im2 = ax2.imshow(hierarchical_pos[0].numpy(), cmap='RdBu_r', aspect='auto')
    ax2.set_title('HierarchicalPositionalEncoding - Positional Embeddings', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(15))
    ax2.set_yticklabels([f'{i}: {name}' for i, name in enumerate(joint_names)], fontsize=8)
    ax2.set_xlabel('Feature Dimension')
    ax2.set_ylabel('Joint')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Plot 3: Joint embedding similarities (Kinematic)
    joint_emb = kinematic_enc.joint_embedding.weight.detach().numpy()
    parent_emb = kinematic_enc.parent_embedding.weight.detach().numpy()
    
    # Compute similarities
    similarities = np.zeros((15, 15))
    for i in range(15):
        for j in range(15):
            similarities[i, j] = np.dot(joint_emb[i], joint_emb[j]) / (np.linalg.norm(joint_emb[i]) * np.linalg.norm(joint_emb[j]))
    
    im3 = ax3.imshow(similarities, cmap='viridis', aspect='auto')
    ax3.set_title('Joint Embedding Similarities (Kinematic)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(15))
    ax3.set_yticks(range(15))
    ax3.set_xticklabels([f'{i}' for i in range(15)], fontsize=8)
    ax3.set_yticklabels([f'{i}' for i in range(15)], fontsize=8)
    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('Joint Index')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Plot 4: Level embedding similarities (Hierarchical)
    level_embs = []
    for emb_layer in hierarchical_enc.level_embeddings:
        level_embs.append(emb_layer.weight.detach().numpy())
    
    # Use first level embedding for similarity
    level_similarities = np.zeros((5, 5))  # 5 levels (0-4)
    for i in range(5):
        for j in range(5):
            similarities[i, j] = np.dot(level_embs[0][i], level_embs[0][j]) / (np.linalg.norm(level_embs[0][i]) * np.linalg.norm(level_embs[0][j]))
    
    im4 = ax4.imshow(similarities[:5, :5], cmap='viridis', aspect='auto')
    ax4.set_title('Level Embedding Similarities (Hierarchical)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(5))
    ax4.set_yticks(range(5))
    ax4.set_xticklabels([f'L{i}' for i in range(5)], fontsize=8)
    ax4.set_yticklabels([f'L{i}' for i in range(5)], fontsize=8)
    ax4.set_xlabel('Level')
    ax4.set_ylabel('Level')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('embedding_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_embedding_flow():
    """Visualize how embeddings flow through the network"""
    device = torch.device('cpu')
    
    # Initialize model
    kinematic_enc = SimpleKinematicEncoding(feature_dim=128)
    
    # Create dummy input
    B, J, F = 1, 15, 128
    joint_features = torch.randn(B, J, F)
    
    with torch.no_grad():
        # Get intermediate embeddings
        joint_ids = torch.arange(J, device=device)
        parent_ids = torch.tensor(kinematic_enc.kinematic_parents, device=device)
        
        joint_emb = kinematic_enc.joint_embedding(joint_ids)  # (15, 64)
        parent_emb = kinematic_enc.parent_embedding(parent_ids)  # (15, 64)
        combined_emb = torch.cat([joint_emb, parent_emb], dim=-1)  # (15, 128)
        pos_encoding = kinematic_enc.fc(combined_emb)  # (15, 128)
        output = joint_features + pos_encoding.unsqueeze(0)  # (B, 15, 128)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Joint embeddings
    im1 = axes[0, 0].imshow(joint_emb.numpy(), cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Joint Embeddings (64D)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Joint Index')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Plot 2: Parent embeddings
    im2 = axes[0, 1].imshow(parent_emb.numpy(), cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Parent Embeddings (64D)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Plot 3: Combined embeddings
    im3 = axes[0, 2].imshow(combined_emb.numpy(), cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('Combined Embeddings (128D)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # Plot 4: Final positional encoding
    im4 = axes[1, 0].imshow(pos_encoding.numpy(), cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('Final Positional Encoding (128D)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Joint Index')
    axes[1, 0].set_xlabel('Feature Dimension')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    # Plot 5: Input features
    im5 = axes[1, 1].imshow(joint_features[0].numpy(), cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('Input Joint Features (128D)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Feature Dimension')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    # Plot 6: Output features
    im6 = axes[1, 2].imshow(output[0].numpy(), cmap='RdBu_r', aspect='auto')
    axes[1, 2].set_title('Output Features (128D)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Feature Dimension')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('embedding_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_embedding_details():
    """Print detailed information about the embeddings"""
    print("🔍 POSITIONAL EMBEDDING ANALYSIS")
    print("=" * 50)
    
    # Joint hierarchy
    joint_names = [
        "Neck", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder",
        "L_Elbow", "L_Wrist", "R_Hip", "R_Knee", "R_Ankle", "R_Foot",
        "L_Hip", "L_Knee", "L_Ankle", "L_Foot"
    ]
    
    kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
    levels = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4]
    
    print("\n📊 JOINT HIERARCHY:")
    print("-" * 30)
    for i, (name, parent, level) in enumerate(zip(joint_names, kinematic_parents, levels)):
        parent_name = joint_names[parent] if parent != i else "Self"
        print(f"Joint {i:2d}: {name:12s} | Parent: {parent:2d} ({parent_name:12s}) | Level: {level}")
    
    print("\n🎯 EMBEDDING COMPONENTS:")
    print("-" * 30)
    print("SimpleKinematicEncoding:")
    print("  • Joint Embedding: 15 joints × 64D = 960 parameters")
    print("  • Parent Embedding: 15 parents × 64D = 960 parameters") 
    print("  • FC Layer: 128D → 128D = 16,512 parameters")
    print("  • Total: ~18,432 parameters")
    
    print("\nHierarchicalPositionalEncoding:")
    print("  • Level Embeddings: 4 layers × 5 levels × 32D = 640 parameters")
    print("  • FC Layer: 128D → 128D = 16,512 parameters")
    print("  • Total: ~17,152 parameters")
    
    print("\n🔄 HOW IT WORKS:")
    print("-" * 30)
    print("1. Input: joint_features (B, 15, 128)")
    print("2. Get joint and parent embeddings")
    print("3. Concatenate: [joint_emb, parent_emb] → (15, 128)")
    print("4. Transform through FC layer")
    print("5. Add to original features: joint_features + pos_encoding")
    print("6. Output: enhanced joint features with kinematic awareness")

if __name__ == "__main__":
    print("🎨 Creating positional embedding visualizations...")
    
    # Create all visualizations
    visualize_kinematic_embedding()
    visualize_embedding_heatmaps()
    visualize_embedding_flow()
    print_embedding_details()
    
    print("\n✅ Visualizations saved as:")
    print("  • kinematic_hierarchy.png")
    print("  • embedding_heatmaps.png") 
    print("  • embedding_flow.png")
