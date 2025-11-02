"""
Evaluation script with attention visualization
Evaluates model and generates comprehensive attention heatmaps

Usage:
    python evaluate_with_attention_viz.py \
        --checkpoint /path/to/best_model_eer_g3_viz_only.pth \
        --database_path /path/to/eval/data \
        --protocols_path /path/to/protocol.txt \
        --track LA \
        --viz_dir attention_analysis \
        --num_viz_samples 100
"""

import argparse
import sys
import torch
from torch.utils.data import DataLoader
from data_utils_SSL import Dataset_ASVspoof2021_eval, Dataset_in_the_wild_eval, genSpoof_list
from model_g3_heatmap import Model
from visualize_attention_evaluation import AttentionVisualizer


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with attention visualization')
    
    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--database_path', type=str, required=True,
                       help='Path to evaluation data directory')
    parser.add_argument('--protocols_path', type=str, required=True,
                       help='Path to protocol file')
    parser.add_argument('--track', type=str, default='LA', 
                       choices=['LA', 'DF', 'In-the-Wild'],
                       help='Evaluation track')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='g3_heatmap')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--group_size', type=int, default=3)
    parser.add_argument('--use_contrastive', action='store_true', default=False)
    
    # Visualization parameters
    parser.add_argument('--viz_dir', type=str, default='attention_analysis',
                       help='Directory to save visualizations')
    parser.add_argument('--num_viz_samples', type=int, default=100,
                       help='Number of samples to analyze for visualization')
    parser.add_argument('--skip_viz', action='store_true', default=False,
                       help='Skip visualization generation')
    parser.add_argument('--classification_viz', action='store_true', default=False,
                       help='Generate classification-based visualizations (correct/incorrect)')
    parser.add_argument('--incorrect_only_viz', action='store_true', default=False,
                       help='Generate ONLY incorrect classification visualizations (False Rejection & False Acceptance)')
    parser.add_argument('--has_labels', action='store_true', default=False,
                       help='Dataset has true labels (for classification analysis)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    print("\n" + "="*70)
    print("📊 EVALUATION WITH ATTENTION VISUALIZATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Track: {args.track}")
    print(f"Database: {args.database_path}")
    print(f"Protocol: {args.protocols_path}")
    print("="*70 + "\n")
    
    # Load evaluation data
    print("📂 Loading evaluation data...")
    file_eval = genSpoof_list(
        dir_meta=args.protocols_path,
        is_train=False,
        is_eval=True
    )
    print(f"   Found {len(file_eval)} evaluation samples")
    
    # Parse labels from protocol file if needed for classification analysis
    label_dict = None
    if args.has_labels:
        print("📋 Parsing labels from protocol file...")
        label_dict = {}
        try:
            with open(args.protocols_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    # Format detection for trial_metadata.txt
                    parsed = False
                    utt_id = None
                    label = None
                    
                    # Format 0A: Bonafide lines - label at column 4
                    # LA_0007-alaw-ita_tx LA_E_5013670-alaw-ita_tx alaw ita_tx bonafide nontarget notrim eval
                    if len(parts) >= 5 and parts[4] == 'bonafide':
                        utt_id = parts[1]  # file_id at column 1
                        label = 1
                        parsed = True
                    
                    # Format 0B: Spoof lines - attack type at column 4, 'spoof' at column 5
                    # LA_0013-alaw-ita_tx LA2021-LA_E_5658320 alaw ita_tx A07 spoof notrim progress
                    elif len(parts) >= 6 and parts[5] == 'spoof':
                        utt_id = parts[1]  # file_id at column 1
                        label = 0
                        parsed = True
                    
                    # Format 0C: Generic - check both column 4 and 5
                    elif len(parts) >= 6:
                        utt_id = parts[1]
                        label_str = parts[4]
                        if label_str == 'bonafide':
                            label = 1
                            parsed = True
                        elif label_str == 'spoof':
                            label = 0
                            parsed = True
                        elif parts[5] in ['bonafide', 'spoof']:
                            label = 1 if parts[5] == 'bonafide' else 0
                            parsed = True
                    
                    # Fallback format: speaker_id file_id - label [attack_type]
                    elif len(parts) >= 4 and parts[2] == '-':
                        utt_id = parts[1]
                        label_str = parts[3]
                        label = 1 if label_str.lower() == 'bonafide' else 0
                        parsed = True
                    
                    # Simple format: file_id label
                    elif len(parts) == 2:
                        utt_id = parts[0]
                        label_str = parts[1]
                        label = 1 if label_str.lower() == 'bonafide' else 0
                        parsed = True
                    
                    if parsed and utt_id is not None:
                        label_dict[utt_id] = label
            print(f"   ✅ Parsed {len(label_dict)} labels")
            bonafide_count = sum(1 for v in label_dict.values() if v == 1)
            spoof_count = len(label_dict) - bonafide_count
            print(f"   📊 Bonafide: {bonafide_count}, Spoof: {spoof_count}")
        except Exception as e:
            print(f"   ⚠️  Failed to parse labels: {e}")
            print(f"   Proceeding without true labels...")
            label_dict = None
            args.has_labels = False
    
    if args.track == 'In-the-Wild':
        eval_set = Dataset_in_the_wild_eval(
            list_IDs=file_eval,
            base_dir=args.database_path
        )
    else:
        eval_set = Dataset_ASVspoof2021_eval(
            list_IDs=file_eval,
            base_dir=args.database_path
        )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=True,  # IMPORTANT: Shuffle to ensure balanced class distribution
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print(f"\n📦 Loading model from checkpoint...")
    model = Model(args, device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"   ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_eer' in checkpoint:
            print(f"   📊 Checkpoint EER: {checkpoint['val_eer']:.2f}%")
    else:
        state_dict = checkpoint
        print(f"   ✅ Loaded checkpoint (legacy format)")
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    # Generate visualizations
    if not args.skip_viz:
        print("\n" + "="*70)
        print("🎨 GENERATING ATTENTION VISUALIZATIONS")
        print("="*70)
        
        visualizer = AttentionVisualizer(model, device, args.viz_dir)
        
        print(f"\n🔍 Collecting attention weights from {args.num_viz_samples} samples...")
        visualizer.collect_attention_weights(
            eval_loader, 
            num_samples=args.num_viz_samples,
            has_labels=args.has_labels,
            label_dict=label_dict,
            balanced_sampling=True  # Always use balanced sampling when labels are available
        )
        
        print("\n📊 Generating visualizations...")
        visualizer.generate_all_visualizations()
        
        # Generate classification-based visualizations if requested
        if args.classification_viz:
            if not args.has_labels:
                print("\n⚠️  Warning: --classification_viz requires --has_labels to be set")
                print("   Skipping classification visualizations")
            else:
                print("\n🎯 Generating classification-based visualizations...")
                visualizer.generate_classification_visualizations(num_samples_per_category=5)
        
        # Generate ONLY incorrect classification visualizations if requested
        if args.incorrect_only_viz:
            if not args.has_labels:
                print("\n⚠️  Warning: --incorrect_only_viz requires --has_labels to be set")
                print("   Skipping incorrect-only visualizations")
            else:
                print("\n❌ Generating incorrect classification visualizations...")
                visualizer.generate_incorrect_only_visualizations(num_samples_per_category=10)
        
        print("\n✅ Visualization complete!")
        print(f"📁 Results saved to: {args.viz_dir}/")
        print("\nGenerated files:")
        print("  📈 layer_weight_heatmap_spoof.png - Layer attention for spoof samples")
        print("  📈 layer_weight_heatmap_bonafide.png - Layer attention for bonafide samples")
        print("  📈 layer_weight_comparison.png - Side-by-side comparison")
        print("  ⏰ temporal_attention_heatmap_*.png - Temporal attention patterns")
        print("  🔗 intra_group_attention_*.png - Intra-group attention weights")
        print("  🌐 inter_group_attention_*.png - Inter-group attention weights")
        
        if args.classification_viz and args.has_labels:
            print("\n  Classification-based visualizations:")
            print("  🎯 layer_weight_correct_bonafide.png - Correctly classified bonafide")
            print("  🎯 layer_weight_incorrect_bonafide.png - Incorrectly classified bonafide (FR)")
            print("  🎯 layer_weight_correct_spoof.png - Correctly classified spoof")
            print("  🎯 layer_weight_incorrect_spoof.png - Incorrectly classified spoof (FA)")
            print("  🎯 classification_comparison_4way.png - 4-way comparison grid")
        
        if args.incorrect_only_viz and args.has_labels:
            print("\n  Incorrect-Only visualizations:")
            print("  ❌ layer_weight_incorrect_bonafide.png - False Rejection (Bonafide → Spoof)")
            print("  ❌ layer_weight_incorrect_spoof.png - False Acceptance (Spoof → Bonafide)")
            print("  ❌ incorrect_classification_comparison.png - Side-by-side error comparison")
    else:
        print("\n⏭️  Skipping visualization (--skip_viz flag set)")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
