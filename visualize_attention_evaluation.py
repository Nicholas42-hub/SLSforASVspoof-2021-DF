"""
Attention Visualization for Evaluation
Creates layer weight heatmaps, temporal attention visualizations, and group-level attention analysis
Similar to Figure 2 in research papers
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path


class AttentionVisualizer:
    """Comprehensive attention visualization for evaluation"""
    
    def __init__(self, model, device, save_dir='attention_viz'):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for attention weights across samples
        self.temporal_attns = []  # Layer-wise temporal attention
        self.intra_attns = []     # Intra-group attention
        self.inter_attns = []     # Inter-group attention
        self.labels = []          # True labels
        self.predictions = []     # Predicted labels
        
    def collect_attention_weights(self, dataloader, num_samples=100, has_labels=False, label_dict=None, balanced_sampling=True):
        """
        Collect attention weights from evaluation samples
        
        Args:
            dataloader: DataLoader with evaluation data
            num_samples: Number of samples to collect (set to None for all)
            has_labels: Whether true labels should be used (requires label_dict)
            label_dict: Dictionary mapping utterance_id to label (0 or 1)
            balanced_sampling: If True, collect equal numbers from each class (requires has_labels=True)
        """
        self.model.eval()
        collected = 0
        
        # For balanced sampling, track per-class collection
        if balanced_sampling and has_labels and label_dict is not None:
            samples_per_class = num_samples // 2 if num_samples is not None else None
            collected_bonafide = 0
            collected_spoof = 0
            print(f"🔍 Collecting attention weights with balanced sampling...")
            print(f"   📊 Target: {samples_per_class} bonafide + {samples_per_class} spoof = {num_samples} total")
        else:
            samples_per_class = None
            print(f"🔍 Collecting attention weights from evaluation samples...")
        
        if has_labels:
            if label_dict is None:
                print(f"   ⚠️  Warning: has_labels=True but no label_dict provided. Using predictions as labels.")
                has_labels = False
                balanced_sampling = False
            else:
                print(f"   ℹ️  True labels available - will track correct/incorrect classifications")
        
        # Enable attention collection mode (set flag on model)
        self.model._collect_attention = True
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader)):
                # Check if we've collected enough samples
                if balanced_sampling and samples_per_class is not None:
                    if collected_bonafide >= samples_per_class and collected_spoof >= samples_per_class:
                        break
                elif num_samples is not None and collected >= num_samples:
                    break
                
                # Evaluation datasets return (audio, utterance_id)
                batch_x, batch_ids = batch_data[0], batch_data[1]
                batch_x = batch_x.to(self.device)
                
                # Get true labels from label_dict if available
                if has_labels and label_dict is not None:
                    true_labels = torch.tensor(
                        [label_dict.get(utt_id, 0) for utt_id in batch_ids],
                        dtype=torch.long,
                        device=self.device
                    )
                else:
                    # Use dummy labels for forward pass
                    true_labels = torch.zeros(batch_x.size(0), dtype=torch.long).to(self.device)
                
                # Forward pass - model will collect attention if _collect_attention flag is set
                try:
                    # Try with return_attention parameter first (newer model versions)
                    output, _, _, _ = self.model.forward(batch_x, labels=true_labels, return_attention=True)
                except TypeError:
                    # Fallback for older model versions without return_attention parameter
                    output, _, _, _ = self.model.forward(batch_x, labels=true_labels)
                
                # Get predictions
                _, preds = output.max(dim=1)
                
                # Check if model has attention_weights attribute
                if not hasattr(self.model, 'attention_weights'):
                    raise AttributeError(
                        "Model does not have 'attention_weights' attribute. "
                        "Please ensure you're using model_g3_heatmap.py with attention collection support."
                    )
                
                # Collect attention weights
                temporal_attn = self.model.attention_weights['temporal']  # (B, L, T)
                intra_attn = self.model.attention_weights['intra']        # (B, num_groups, group_size)
                inter_attn = self.model.attention_weights['inter']        # (B, num_groups)
                
                # Check if attention weights were actually collected
                if temporal_attn is None:
                    raise ValueError(
                        "Temporal attention is None. The model may not be collecting attention weights. "
                        "Check that the model's forward method stores attention in self.attention_weights."
                    )
                
                # Store batch results with balanced sampling
                for i in range(batch_x.size(0)):
                    # Determine sample label
                    if has_labels and label_dict is not None:
                        sample_label = true_labels[i].cpu().item()
                    else:
                        sample_label = preds[i].cpu().item()
                    
                    # For balanced sampling, check if we need this class
                    if balanced_sampling and samples_per_class is not None:
                        if sample_label == 1 and collected_bonafide >= samples_per_class:
                            continue  # Skip, already have enough bonafide
                        if sample_label == 0 and collected_spoof >= samples_per_class:
                            continue  # Skip, already have enough spoof
                    elif num_samples is not None and collected >= num_samples:
                        break
                    
                    # Store sample
                    self.temporal_attns.append(temporal_attn[i].cpu().numpy())
                    if intra_attn is not None:
                        self.intra_attns.append(intra_attn[i].cpu().numpy())
                    if inter_attn is not None:
                        self.inter_attns.append(inter_attn[i].cpu().numpy())
                    
                    # Store true label and prediction
                    self.labels.append(sample_label)
                    self.predictions.append(preds[i].cpu().item())
                    
                    # Update counters
                    collected += 1
                    if balanced_sampling and samples_per_class is not None:
                        if sample_label == 1:
                            collected_bonafide += 1
                        else:
                            collected_spoof += 1
        
        # Disable attention collection mode
        self.model._collect_attention = False
        
        print(f"✅ Collected attention weights from {collected} samples")
        if balanced_sampling and samples_per_class is not None:
            print(f"   📊 Bonafide: {collected_bonafide}, Spoof: {collected_spoof}")
        print(f"   - Temporal attention shape: {self.temporal_attns[0].shape}")
        if self.intra_attns:
            print(f"   - Intra-group attention shape: {self.intra_attns[0].shape}")
        if self.inter_attns:
            print(f"   - Inter-group attention shape: {self.inter_attns[0].shape}")
        
        # Print classification statistics if we have true labels
        if has_labels:
            self._print_classification_stats()
    
    def _print_classification_stats(self):
        """Print classification accuracy statistics"""
        labels_arr = np.array(self.labels)
        preds_arr = np.array(self.predictions)
        
        # Overall accuracy
        accuracy = np.mean(labels_arr == preds_arr)
        
        # Per-class statistics
        bonafide_mask = labels_arr == 1
        spoof_mask = labels_arr == 0
        
        bonafide_correct = np.sum((labels_arr == 1) & (preds_arr == 1))
        bonafide_total = np.sum(bonafide_mask)
        bonafide_incorrect = bonafide_total - bonafide_correct
        
        spoof_correct = np.sum((labels_arr == 0) & (preds_arr == 0))
        spoof_total = np.sum(spoof_mask)
        spoof_incorrect = spoof_total - spoof_correct
        
        print(f"\n📊 Classification Statistics:")
        print(f"   Overall Accuracy: {accuracy*100:.2f}% ({np.sum(labels_arr == preds_arr)}/{len(labels_arr)})")
        print(f"\n   Bonafide (label=1):")
        if bonafide_total > 0:
            print(f"      ✅ Correct: {bonafide_correct}/{bonafide_total} ({bonafide_correct/bonafide_total*100:.2f}%)")
            print(f"      ❌ Incorrect (False Rejection): {bonafide_incorrect}/{bonafide_total} ({bonafide_incorrect/bonafide_total*100:.2f}%)")
        else:
            print(f"      ⚠️  No bonafide samples collected")
        print(f"\n   Spoof (label=0):")
        if spoof_total > 0:
            print(f"      ✅ Correct: {spoof_correct}/{spoof_total} ({spoof_correct/spoof_total*100:.2f}%)")
            print(f"      ❌ Incorrect (False Acceptance): {spoof_incorrect}/{spoof_total} ({spoof_incorrect/spoof_total*100:.2f}%)")
        else:
            print(f"      ⚠️  No spoof samples collected")
    
    def _get_category_indices(self, category):
        """
        Get indices for a specific classification category
        
        Args:
            category: One of 'correct_bonafide', 'incorrect_bonafide', 'correct_spoof', 'incorrect_spoof'
        
        Returns:
            List of indices matching the category
        """
        labels_arr = np.array(self.labels)
        preds_arr = np.array(self.predictions)
        
        if category == 'correct_bonafide':
            # True label=1, Prediction=1
            mask = (labels_arr == 1) & (preds_arr == 1)
        elif category == 'incorrect_bonafide':
            # True label=1, Prediction=0 (False Rejection)
            mask = (labels_arr == 1) & (preds_arr == 0)
        elif category == 'correct_spoof':
            # True label=0, Prediction=0
            mask = (labels_arr == 0) & (preds_arr == 0)
        elif category == 'incorrect_spoof':
            # True label=0, Prediction=1 (False Acceptance)
            mask = (labels_arr == 0) & (preds_arr == 1)
        else:
            raise ValueError(f"Unknown category: {category}")
        
        return np.where(mask)[0].tolist()
    
    def plot_layer_weight_heatmap(self, num_runs=5, class_label=None):
        """
        Create layer weight heatmap similar to Figure 2(a) and 2(b)
        Shows attention weights across layers for different model runs (samples)
        
        Args:
            num_runs: Number of random samples to display as "model runs"
            class_label: Filter by class (0=spoof, 1=bonafide, None=all)
        """
        if not self.temporal_attns:
            print("❌ No attention weights collected. Run collect_attention_weights() first.")
            return
        
        # Filter by class if specified
        if class_label is not None:
            indices = [i for i, label in enumerate(self.labels) if label == class_label]
            class_name = "Bonafide" if class_label == 1 else "Spoof"
        else:
            indices = list(range(len(self.temporal_attns)))
            class_name = "All"
        
        # Skip if no samples available for this class
        if len(indices) == 0:
            print(f"⚠️  No {class_name} samples available - skipping visualization")
            return
        
        if len(indices) < num_runs:
            num_runs = len(indices)
            print(f"⚠️  Only {num_runs} {class_name} samples available")
        
        # Randomly sample runs
        np.random.seed(42)
        selected_indices = np.random.choice(indices, size=num_runs, replace=False)
        
        # Compute layer-wise importance by averaging temporal attention
        # temporal_attns[i] has shape (L, T) - average over T to get layer importance
        layer_importances = []
        for idx in selected_indices:
            temporal_attn = self.temporal_attns[idx]  # (L, T)
            layer_importance = temporal_attn.mean(axis=1)  # (L,)
            layer_importances.append(layer_importance)
        
        # Stack into matrix: (num_runs, num_layers)
        layer_weight_matrix = np.stack(layer_importances, axis=0)  # (num_runs, L)
        
        # Transpose to have layers on x-axis: (L, num_runs)
        layer_weight_matrix = layer_weight_matrix.T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 5))
        
        sns.heatmap(
            layer_weight_matrix,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=[f'Run {i+1}' for i in range(num_runs)],
            yticklabels=range(layer_weight_matrix.shape[0]),
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        
        ax.set_xlabel('Model Run', fontsize=14, fontweight='bold')
        ax.set_ylabel('Layer', fontsize=14, fontweight='bold')
        ax.set_title(f'Layer Weight Heatmap - {class_name} Samples', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / f'layer_weight_heatmap_{class_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved layer weight heatmap: {save_path}")
        plt.close()
    
    def plot_temporal_attention_heatmap(self, num_samples=10, class_label=None):
        """
        Visualize temporal attention patterns across layers
        Shows how attention is distributed across time frames for different layers
        
        Args:
            num_samples: Number of samples to average
            class_label: Filter by class (0=spoof, 1=bonafide, None=all)
        """
        if not self.temporal_attns:
            print("❌ No attention weights collected.")
            return
        
        # Filter by class
        if class_label is not None:
            indices = [i for i, label in enumerate(self.labels) if label == class_label]
            class_name = "Bonafide" if class_label == 1 else "Spoof"
        else:
            indices = list(range(len(self.temporal_attns)))
            class_name = "All"
        
        # Skip if no samples available
        if len(indices) == 0:
            print(f"⚠️  No {class_name} samples available - skipping temporal attention visualization")
            return
        
        if len(indices) < num_samples:
            num_samples = len(indices)
        
        # Average temporal attention across selected samples
        selected_indices = np.random.choice(indices, size=num_samples, replace=False)
        temporal_attns_selected = [self.temporal_attns[i] for i in selected_indices]
        
        # Average: (num_samples, L, T) -> (L, T)
        avg_temporal_attn = np.mean(np.stack(temporal_attns_selected, axis=0), axis=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(
            avg_temporal_attn,
            cmap='viridis',  # Consistent color scheme
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=False,  # Too many time frames to show all
            yticklabels=range(avg_temporal_attn.shape[0]),
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        
        ax.set_xlabel('Time Frame', fontsize=14, fontweight='bold')
        ax.set_ylabel('Layer', fontsize=14, fontweight='bold')
        ax.set_title(f'Temporal Attention Heatmap - {class_name} Samples\nAveraged over {num_samples} samples', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / f'temporal_attention_heatmap_{class_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved temporal attention heatmap: {save_path}")
        plt.close()
    
    def plot_intra_group_attention(self, class_label=None):
        """
        Visualize intra-group attention patterns
        Shows how layers within each group attend to each other
        
        Args:
            class_label: Filter by class (0=spoof, 1=bonafide, None=all)
        """
        if not self.intra_attns:
            print("❌ No intra-group attention weights collected.")
            return
        
        # Filter by class
        if class_label is not None:
            indices = [i for i, label in enumerate(self.labels) if label == class_label]
            class_name = "Bonafide" if class_label == 1 else "Spoof"
        else:
            indices = list(range(len(self.intra_attns)))
            class_name = "All"
        
        # Skip if no samples available
        if len(indices) == 0:
            print(f"⚠️  No {class_name} samples available - skipping intra-group attention visualization")
            return
        
        # Average intra attention across samples: (num_samples, num_groups, group_size) -> (num_groups, group_size)
        intra_attns_selected = [self.intra_attns[i] for i in indices]
        avg_intra_attn = np.mean(np.stack(intra_attns_selected, axis=0), axis=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(
            avg_intra_attn,
            cmap='viridis',  # Consistent color scheme
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=[f'L{i}' for i in range(avg_intra_attn.shape[1])],
            yticklabels=[f'Group {i+1}' for i in range(avg_intra_attn.shape[0])],
            annot=avg_intra_attn.shape[0] <= 10,  # Show values if not too many groups
            fmt='.3f',
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        
        ax.set_xlabel('Layer within Group', fontsize=14, fontweight='bold')
        ax.set_ylabel('Group', fontsize=14, fontweight='bold')
        ax.set_title(f'Intra-Group Attention - {class_name} Samples\nAveraged over {len(indices)} samples', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / f'intra_group_attention_{class_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved intra-group attention: {save_path}")
        plt.close()
    
    def plot_inter_group_attention(self, class_label=None):
        """
        Visualize inter-group attention patterns as a heatmap
        Shows how different groups of layers are weighted
        
        Args:
            class_label: Filter by class (0=spoof, 1=bonafide, None=all)
        """
        if not self.inter_attns:
            print("❌ No inter-group attention weights collected.")
            return
        
        # Filter by class
        if class_label is not None:
            indices = [i for i, label in enumerate(self.labels) if label == class_label]
            class_name = "Bonafide" if class_label == 1 else "Spoof"
        else:
            indices = list(range(len(self.inter_attns)))
            class_name = "All"
        
        # Skip if no samples available
        if len(indices) == 0:
            print(f"⚠️  No {class_name} samples available - skipping inter-group attention visualization")
            return
        
        # Average inter attention across samples: (num_samples, num_groups) -> (num_groups,)
        inter_attns_selected = [self.inter_attns[i] for i in indices]
        avg_inter_attn = np.mean(np.stack(inter_attns_selected, axis=0), axis=0)
        
        # Reshape to 2D for heatmap: (1, num_groups)
        inter_attn_2d = avg_inter_attn.reshape(1, -1)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 3))
        
        sns.heatmap(
            inter_attn_2d,
            cmap='viridis',  # Consistent color scheme
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=[f'Group {i+1}' for i in range(len(avg_inter_attn))],
            yticklabels=['Inter-Group'],
            annot=True,
            fmt='.3f',
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        
        ax.set_xlabel('Layer Group', fontsize=14, fontweight='bold')
        ax.set_title(f'Inter-Group Attention - {class_name} Samples\nAveraged over {len(indices)} samples', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / f'inter_group_attention_{class_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved inter-group attention: {save_path}")
        plt.close()
    
    def plot_comparison_heatmap(self, num_runs_per_class=5):
        """
        Create comparison heatmap showing spoof vs bonafide samples
        Similar to Figure 2 with multiple subfigures
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        for class_idx, (class_label, class_name, ax) in enumerate([
            (0, 'Spoof', axes[0]),
            (1, 'Bonafide', axes[1])
        ]):
            # Get samples for this class
            indices = [i for i, label in enumerate(self.labels) if label == class_label]
            
            if len(indices) < num_runs_per_class:
                num_runs_per_class_actual = len(indices)
            else:
                num_runs_per_class_actual = num_runs_per_class
            
            # Sample runs
            selected_indices = np.random.choice(indices, size=num_runs_per_class_actual, replace=False)
            
            # Compute layer importance
            layer_importances = []
            for idx in selected_indices:
                temporal_attn = self.temporal_attns[idx]  # (L, T)
                layer_importance = temporal_attn.mean(axis=1)  # (L,)
                layer_importances.append(layer_importance)
            
            layer_weight_matrix = np.stack(layer_importances, axis=0)
            
            # Transpose to have layers on x-axis: (L, num_samples)
            layer_weight_matrix = layer_weight_matrix.T
            
            # Plot heatmap
            sns.heatmap(
                layer_weight_matrix,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=[f'Sample {i+1}' for i in range(num_runs_per_class_actual)],
                yticklabels=range(layer_weight_matrix.shape[0]),
                linewidths=0.5,
                linecolor='white',
                ax=ax
            )
            
            ax.set_xlabel('Sample', fontsize=12, fontweight='bold')
            ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
            ax.set_title(f'({chr(97+class_idx)}) Layer weight heatmap for {class_name} samples', 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / 'layer_weight_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved comparison heatmap: {save_path}")
        plt.close()
    
    def plot_classification_category_heatmap(self, category, num_samples=5):
        """
        Create layer weight heatmap for a specific classification category
        
        Args:
            category: One of 'correct_bonafide', 'incorrect_bonafide', 'correct_spoof', 'incorrect_spoof'
            num_samples: Number of samples to display
        """
        if not self.temporal_attns:
            print("❌ No attention weights collected.")
            return
        
        # Get indices for this category
        indices = self._get_category_indices(category)
        
        # Category name mapping
        category_names = {
            'correct_bonafide': 'Correctly Classified Bonafide',
            'incorrect_bonafide': 'Incorrectly Classified Bonafide (False Rejection)',
            'correct_spoof': 'Correctly Classified Spoof',
            'incorrect_spoof': 'Incorrectly Classified Spoof (False Acceptance)'
        }
        category_name = category_names.get(category, category)
        
        if len(indices) == 0:
            print(f"⚠️  No samples found for category: {category_name}")
            return
        
        if len(indices) < num_samples:
            num_samples = len(indices)
            print(f"⚠️  Only {num_samples} samples available for {category_name}")
        
        # Randomly sample
        np.random.seed(42)
        selected_indices = np.random.choice(indices, size=num_samples, replace=False)
        
        # Compute layer importance
        layer_importances = []
        for idx in selected_indices:
            temporal_attn = self.temporal_attns[idx]  # (L, T)
            layer_importance = temporal_attn.mean(axis=1)  # (L,)
            layer_importances.append(layer_importance)
        
        layer_weight_matrix = np.stack(layer_importances, axis=0)  # (num_samples, L)
        
        # Transpose to have layers on x-axis: (L, num_samples)
        layer_weight_matrix = layer_weight_matrix.T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 5))
        
        sns.heatmap(
            layer_weight_matrix,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=[f'Sample {i+1}' for i in range(num_samples)],
            yticklabels=range(layer_weight_matrix.shape[0]),
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        
        ax.set_xlabel('Sample', fontsize=14, fontweight='bold')
        ax.set_ylabel('Layer', fontsize=14, fontweight='bold')
        ax.set_title(f'{category_name}\n({len(indices)} samples available, showing {num_samples})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / f'layer_weight_{category}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved {category_name} heatmap: {save_path}")
        plt.close()
    
    def plot_classification_comparison(self, num_samples_per_category=5):
        """
        Create a 2x2 grid comparing all four classification categories
        """
        categories = [
            ('correct_bonafide', 'Correct Bonafide'),
            ('incorrect_bonafide', 'Incorrect Bonafide (FR)'),
            ('correct_spoof', 'Correct Spoof'),
            ('incorrect_spoof', 'Incorrect Spoof (FA)')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (category, title) in enumerate(categories):
            ax = axes[idx]
            
            # Get indices for this category
            indices = self._get_category_indices(category)
            
            if len(indices) == 0:
                ax.text(0.5, 0.5, f'No samples\nfor {title}', 
                       ha='center', va='center', fontsize=14)
                ax.set_title(f'({chr(97+idx)}) {title}', fontsize=12, fontweight='bold')
                continue
            
            num_samples = min(num_samples_per_category, len(indices))
            
            # Sample and compute layer importance
            np.random.seed(42)
            selected_indices = np.random.choice(indices, size=num_samples, replace=False)
            
            layer_importances = []
            for sample_idx in selected_indices:
                temporal_attn = self.temporal_attns[sample_idx]
                layer_importance = temporal_attn.mean(axis=1)
                layer_importances.append(layer_importance)
            
            layer_weight_matrix = np.stack(layer_importances, axis=0).T  # (L, num_samples)
            
            # Plot heatmap
            sns.heatmap(
                layer_weight_matrix,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=[f'S{i+1}' for i in range(num_samples)],
                yticklabels=range(layer_weight_matrix.shape[0]) if idx % 2 == 0 else False,
                linewidths=0.5,
                linecolor='white',
                ax=ax
            )
            
            ax.set_xlabel('Sample', fontsize=11, fontweight='bold')
            if idx % 2 == 0:
                ax.set_ylabel('Layer', fontsize=11, fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {title} (n={len(indices)})', 
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / 'classification_comparison_4way.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved 4-way classification comparison: {save_path}")
        plt.close()
    
    def generate_all_visualizations(self, num_samples_collect=100):
        """
        Generate all attention visualizations in one go
        
        Args:
            num_samples_collect: Number of samples to collect for analysis
        """
        print("\n" + "="*70)
        print("🎨 GENERATING ATTENTION VISUALIZATIONS")
        print("="*70)
        
        # Check class balance
        labels_arr = np.array(self.labels)
        num_bonafide = np.sum(labels_arr == 1)
        num_spoof = np.sum(labels_arr == 0)
        
        if num_bonafide == 0 or num_spoof == 0:
            print(f"\n⚠️  WARNING: Imbalanced dataset detected!")
            print(f"   Bonafide samples: {num_bonafide}")
            print(f"   Spoof samples: {num_spoof}")
            print(f"   Some visualizations will be skipped.\n")
        
        # 1. Layer weight heatmaps (like Figure 2)
        print("\n📊 Generating layer weight heatmaps...")
        self.plot_layer_weight_heatmap(num_runs=5, class_label=0)  # Spoof
        self.plot_layer_weight_heatmap(num_runs=5, class_label=1)  # Bonafide
        
        # Only generate comparison if both classes are present
        if num_bonafide > 0 and num_spoof > 0:
            self.plot_comparison_heatmap(num_runs_per_class=5)
        else:
            print("⚠️  Skipping comparison heatmap - need both classes")
        
        # 2. Temporal attention visualization
        print("\n⏰ Generating temporal attention heatmaps...")
        self.plot_temporal_attention_heatmap(num_samples=200, class_label=0)  # Spoof
        self.plot_temporal_attention_heatmap(num_samples=200, class_label=1)  # Bonafide
        
        # 3. Intra-group attention heatmaps
        print("\n🔗 Generating intra-group attention heatmaps...")
        self.plot_intra_group_attention(class_label=0)  # Spoof
        self.plot_intra_group_attention(class_label=1)  # Bonafide
        
        # 4. Inter-group attention heatmaps
        print("\n🌐 Generating inter-group attention heatmaps...")
        self.plot_inter_group_attention(class_label=0)  # Spoof
        self.plot_inter_group_attention(class_label=1)  # Bonafide
        
        print("\n" + "="*70)
        print(f"✅ ALL VISUALIZATIONS SAVED TO: {self.save_dir}")
        print("="*70 + "\n")
    
    def generate_classification_visualizations(self, num_samples_per_category=5):
        """
        Generate visualizations based on classification correctness
        Requires that true labels were provided during collection
        
        Args:
            num_samples_per_category: Number of samples to show per category
        """
        # Check if we have different labels and predictions (indicates true labels were used)
        labels_arr = np.array(self.labels)
        preds_arr = np.array(self.predictions)
        
        if np.array_equal(labels_arr, preds_arr):
            print("⚠️  Warning: Labels and predictions are identical.")
            print("   This suggests no true labels were provided during collection.")
            print("   Use collect_attention_weights(..., has_labels=True) with a labeled dataset.")
            return
        
        print("\n" + "="*70)
        print("🎯 GENERATING CLASSIFICATION-BASED VISUALIZATIONS")
        print("="*70)
        
        # 1. Individual category heatmaps
        print("\n📊 Generating individual category heatmaps...")
        for category in ['correct_bonafide', 'incorrect_bonafide', 'correct_spoof', 'incorrect_spoof']:
            self.plot_classification_category_heatmap(category, num_samples=num_samples_per_category)
        
        # 2. 4-way comparison
        print("\n🔍 Generating 4-way classification comparison...")
        self.plot_classification_comparison(num_samples_per_category=num_samples_per_category)
        
        print("\n" + "="*70)
        print(f"✅ CLASSIFICATION VISUALIZATIONS SAVED TO: {self.save_dir}")
        print("="*70 + "\n")
    
    def generate_incorrect_only_visualizations(self, num_samples_per_category=10):
        """
        Generate visualizations ONLY for incorrectly classified samples
        - Incorrect Bonafide (False Rejection): True bonafide misclassified as spoof
        - Incorrect Spoof (False Acceptance): True spoof misclassified as bonafide
        
        Args:
            num_samples_per_category: Number of samples to show per incorrect category
        """
        # Check if we have different labels and predictions
        labels_arr = np.array(self.labels)
        preds_arr = np.array(self.predictions)
        
        if np.array_equal(labels_arr, preds_arr):
            print("⚠️  Warning: All predictions are correct (100% accuracy).")
            print("   No incorrectly classified samples to visualize.")
            return
        
        print("\n" + "="*70)
        print("❌ GENERATING INCORRECT CLASSIFICATION VISUALIZATIONS")
        print("="*70)
        
        # Get error statistics
        incorrect_bonafide_idx = self._get_category_indices('incorrect_bonafide')
        incorrect_spoof_idx = self._get_category_indices('incorrect_spoof')
        
        print(f"\n📊 Error Analysis:")
        print(f"   False Rejections (Bonafide → Spoof): {len(incorrect_bonafide_idx)} samples")
        print(f"   False Acceptances (Spoof → Bonafide): {len(incorrect_spoof_idx)} samples")
        
        # 1. Individual incorrect category heatmaps
        print("\n📊 Generating individual error heatmaps...")
        self.plot_classification_category_heatmap('incorrect_bonafide', num_samples=num_samples_per_category)
        self.plot_classification_category_heatmap('incorrect_spoof', num_samples=num_samples_per_category)
        
        # 2. Side-by-side comparison of incorrect classifications
        print("\n🔍 Generating incorrect classification comparison...")
        self.plot_incorrect_comparison(num_samples_per_category=num_samples_per_category)
        
        print("\n" + "="*70)
        print(f"✅ INCORRECT CLASSIFICATION VISUALIZATIONS SAVED TO: {self.save_dir}")
        print("="*70 + "\n")
    
    def plot_incorrect_comparison(self, num_samples_per_category=10):
        """
        Create a 1x2 comparison showing only incorrect classifications
        Left: False Rejection (Bonafide misclassified as Spoof)
        Right: False Acceptance (Spoof misclassified as Bonafide)
        """
        categories = [
            ('incorrect_bonafide', 'False Rejection\n(Bonafide → Spoof)'),
            ('incorrect_spoof', 'False Acceptance\n(Spoof → Bonafide)')
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for idx, (category, title) in enumerate(categories):
            ax = axes[idx]
            
            # Get indices for this category
            indices = self._get_category_indices(category)
            
            if len(indices) == 0:
                ax.text(0.5, 0.5, f'No samples\nfor {title}', 
                       ha='center', va='center', fontsize=14)
                ax.set_title(f'({chr(97+idx)}) {title}', fontsize=14, fontweight='bold')
                ax.axis('off')
                continue
            
            num_samples = min(num_samples_per_category, len(indices))
            
            # Sample and compute layer importance
            np.random.seed(42)
            selected_indices = np.random.choice(indices, size=num_samples, replace=False)
            
            layer_importances = []
            for sample_idx in selected_indices:
                temporal_attn = self.temporal_attns[sample_idx]
                layer_importance = temporal_attn.mean(axis=1)
                layer_importances.append(layer_importance)
            
            layer_weight_matrix = np.stack(layer_importances, axis=0).T  # (L, num_samples)
            
            # Plot heatmap
            sns.heatmap(
                layer_weight_matrix,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=[f'S{i+1}' for i in range(num_samples)],
                yticklabels=range(layer_weight_matrix.shape[0]),
                linewidths=0.5,
                linecolor='white',
                ax=ax
            )
            
            ax.set_xlabel('Sample', fontsize=13, fontweight='bold')
            ax.set_ylabel('Layer', fontsize=13, fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {title}\n(n={len(indices)}, showing {num_samples})', 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / 'incorrect_classification_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved incorrect classification comparison: {save_path}")
        plt.close()


def visualize_attention_from_checkpoint(
    checkpoint_path,
    eval_loader,
    device,
    save_dir='attention_visualizations',
    num_samples=100
):
    """
    Main function to generate all attention visualizations from a checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        eval_loader: DataLoader with evaluation data
        device: torch device
        save_dir: Directory to save visualizations
        num_samples: Number of samples to analyze
    
    Example usage:
        from model_g3_heatmap import Model
        
        # Load model
        model = Model(args, device)
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        
        # Generate visualizations
        visualize_attention_from_checkpoint(
            checkpoint_path,
            eval_loader,
            device,
            save_dir='viz_best_model'
        )
    """
    from model_g3_heatmap import Model
    
    # Load model (assuming args are provided globally or you modify this)
    print(f"📦 Loading model from: {checkpoint_path}")
    # Note: You'll need to pass args here or load them from checkpoint
    # This is a simplified version - adapt to your needs
    
    print("🎨 Creating attention visualizer...")
    visualizer = AttentionVisualizer(model, device, save_dir)
    
    print(f"🔍 Collecting attention weights from {num_samples} samples...")
    visualizer.collect_attention_weights(eval_loader, num_samples=num_samples)
    
    print("🎨 Generating all visualizations...")
    visualizer.generate_all_visualizations()
    
    return visualizer


if __name__ == '__main__':
    """
    Example standalone usage
    """
    import argparse
    from torch.utils.data import DataLoader
    from data_utils_SSL import Dataset_ASVspoof2021_eval, genSpoof_list
    from model_g3_heatmap import Model
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--database_path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--protocols_path', type=str, required=True, help='Path to protocol file')
    parser.add_argument('--save_dir', type=str, default='attention_viz', help='Save directory')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--group_size', type=int, default=3, help='Group size for model')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load evaluation data
    file_eval = genSpoof_list(dir_meta=args.protocols_path, is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=args.database_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = Model(args, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create visualizer and generate plots
    visualizer = AttentionVisualizer(model, device, args.save_dir)
    visualizer.collect_attention_weights(eval_loader, num_samples=args.num_samples)
    visualizer.generate_all_visualizations()
