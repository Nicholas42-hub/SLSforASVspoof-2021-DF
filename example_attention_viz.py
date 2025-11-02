"""
Example: Generate attention visualizations programmatically
This script demonstrates how to use the AttentionVisualizer API directly
"""

import torch
from torch.utils.data import DataLoader
from model_g3_heatmap import Model
from data_utils_SSL import Dataset_ASVspoof2021_eval, genSpoof_list
from visualize_attention_evaluation import AttentionVisualizer


def create_simple_args(group_size=3):
    """Create minimal args object for model initialization"""
    class Args:
        def __init__(self):
            self.group_size = group_size
            self.use_contrastive = False
            self.contrastive_weight = 0.1
            self.supcon_weight = 0.1
    return Args()


def example_1_basic_usage():
    """
    Example 1: Basic usage - Generate all visualizations
    """
    print("\n" + "="*70)
    print("Example 1: Basic Attention Visualization")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'models/best_model_eer_g3_viz_only.pth'
    
    # Load data
    file_eval = genSpoof_list(
        dir_meta='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt',
        is_train=False,
        is_eval=True
    )
    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_eval,
        base_dir='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/'
    )
    eval_loader = DataLoader(eval_set, batch_size=16, shuffle=False, num_workers=4)
    
    # Load model
    args = create_simple_args(group_size=3)
    model = Model(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create visualizer and generate all plots
    visualizer = AttentionVisualizer(model, device, save_dir='attention_basic_example')
    visualizer.collect_attention_weights(eval_loader, num_samples=100)
    visualizer.generate_all_visualizations()
    
    print("✅ Example 1 complete! Check 'attention_basic_example/' directory")


def example_2_custom_visualizations():
    """
    Example 2: Generate specific visualizations with custom parameters
    """
    print("\n" + "="*70)
    print("Example 2: Custom Visualizations")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'models/best_model_eer_g3_viz_only.pth'
    
    # Load data (shortened for example)
    file_eval = genSpoof_list(
        dir_meta='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt',
        is_train=False,
        is_eval=True
    )
    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_eval,
        base_dir='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/'
    )
    eval_loader = DataLoader(eval_set, batch_size=16, shuffle=False, num_workers=4)
    
    # Load model
    args = create_simple_args(group_size=3)
    model = Model(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, device, save_dir='attention_custom_example')
    visualizer.collect_attention_weights(eval_loader, num_samples=50)
    
    # Generate only specific visualizations
    print("Generating layer weight heatmap for spoof samples only...")
    visualizer.plot_layer_weight_heatmap(num_runs=10, class_label=0)  # 0 = spoof
    
    print("Generating temporal attention for bonafide samples only...")
    visualizer.plot_temporal_attention_heatmap(num_samples=30, class_label=1)  # 1 = bonafide
    
    print("Generating comparison heatmap...")
    visualizer.plot_comparison_heatmap(num_runs_per_class=7)
    
    print("✅ Example 2 complete! Check 'attention_custom_example/' directory")


def example_3_analyze_attention_data():
    """
    Example 3: Access and analyze raw attention data
    """
    print("\n" + "="*70)
    print("Example 3: Analyzing Raw Attention Data")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'models/best_model_eer_g3_viz_only.pth'
    
    # Load data
    file_eval = genSpoof_list(
        dir_meta='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt',
        is_train=False,
        is_eval=True
    )
    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_eval,
        base_dir='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/'
    )
    eval_loader = DataLoader(eval_set, batch_size=16, shuffle=False, num_workers=4)
    
    # Load model
    args = create_simple_args(group_size=3)
    model = Model(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint)
    model = model.to(device)
    model.eval()
    
    # Collect attention weights
    visualizer = AttentionVisualizer(model, device, save_dir='attention_analysis_example')
    visualizer.collect_attention_weights(eval_loader, num_samples=100)
    
    # Access raw attention data
    print(f"\n📊 Collected data from {len(visualizer.temporal_attns)} samples")
    print(f"   Temporal attention shape: {visualizer.temporal_attns[0].shape}")
    if visualizer.intra_attns:
        print(f"   Intra-group attention shape: {visualizer.intra_attns[0].shape}")
    if visualizer.inter_attns:
        print(f"   Inter-group attention shape: {visualizer.inter_attns[0].shape}")
    
    # Analyze specific samples
    import numpy as np
    
    # Find spoof and bonafide samples
    spoof_indices = [i for i, label in enumerate(visualizer.labels) if label == 0]
    bonafide_indices = [i for i, label in enumerate(visualizer.labels) if label == 1]
    
    print(f"\n🔍 Sample distribution:")
    print(f"   Spoof samples: {len(spoof_indices)}")
    print(f"   Bonafide samples: {len(bonafide_indices)}")
    
    # Compute average layer importance for each class
    spoof_layer_importance = np.mean([
        visualizer.temporal_attns[i].mean(axis=1) for i in spoof_indices
    ], axis=0)
    
    bonafide_layer_importance = np.mean([
        visualizer.temporal_attns[i].mean(axis=1) for i in bonafide_indices
    ], axis=0)
    
    print(f"\n📈 Average layer importance:")
    print(f"   Spoof - Top 3 layers: {np.argsort(spoof_layer_importance)[-3:][::-1]}")
    print(f"   Bonafide - Top 3 layers: {np.argsort(bonafide_layer_importance)[-3:][::-1]}")
    
    # Compute inter-group attention statistics
    if visualizer.inter_attns:
        spoof_inter = np.mean([visualizer.inter_attns[i] for i in spoof_indices], axis=0)
        bonafide_inter = np.mean([visualizer.inter_attns[i] for i in bonafide_indices], axis=0)
        
        print(f"\n🌐 Inter-group attention:")
        print(f"   Spoof - Most important group: {np.argmax(spoof_inter)} (weight: {np.max(spoof_inter):.4f})")
        print(f"   Bonafide - Most important group: {np.argmax(bonafide_inter)} (weight: {np.max(bonafide_inter):.4f})")
    
    print("\n✅ Example 3 complete! Raw attention data analyzed")


def example_4_compare_checkpoints():
    """
    Example 4: Compare attention patterns across different checkpoints
    """
    print("\n" + "="*70)
    print("Example 4: Compare Multiple Checkpoints")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoints = [
        ('models/checkpoint_epoch_5.pth', 'epoch_5'),
        ('models/checkpoint_epoch_10.pth', 'epoch_10'),
        ('models/best_model_eer_g3_viz_only.pth', 'best_model')
    ]
    
    # Load data once
    file_eval = genSpoof_list(
        dir_meta='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt',
        is_train=False,
        is_eval=True
    )
    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_eval,
        base_dir='/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/'
    )
    eval_loader = DataLoader(eval_set, batch_size=16, shuffle=False, num_workers=4)
    
    args = create_simple_args(group_size=3)
    
    for checkpoint_path, name in checkpoints:
        print(f"\n📦 Processing {name}...")
        
        # Load model
        model = Model(args, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint)
        model = model.to(device)
        model.eval()
        
        # Generate visualizations in separate directory
        visualizer = AttentionVisualizer(model, device, save_dir=f'attention_compare/{name}')
        visualizer.collect_attention_weights(eval_loader, num_samples=50)
        visualizer.generate_all_visualizations()
        
        print(f"   ✅ {name} visualizations saved to attention_compare/{name}/")
    
    print("\n✅ Example 4 complete! Compare visualizations in 'attention_compare/' subdirectories")


if __name__ == '__main__':
    """
    Run examples
    Uncomment the example you want to run
    """
    
    # Example 1: Basic usage - generate all visualizations
    # example_1_basic_usage()
    
    # Example 2: Custom visualizations - generate specific plots
    # example_2_custom_visualizations()
    
    # Example 3: Analyze raw attention data
    # example_3_analyze_attention_data()
    
    # Example 4: Compare multiple checkpoints
    # example_4_compare_checkpoints()
    
    print("\n" + "="*70)
    print("💡 Uncomment the example you want to run in this script")
    print("="*70)
    print("\nAvailable examples:")
    print("  1. example_1_basic_usage() - Generate all visualizations")
    print("  2. example_2_custom_visualizations() - Custom plots")
    print("  3. example_3_analyze_attention_data() - Analyze raw data")
    print("  4. example_4_compare_checkpoints() - Compare models")
    print("="*70 + "\n")
