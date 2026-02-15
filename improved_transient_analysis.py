"""
改进的瞬态分析方法

问题：原方法使用激活值的平均与标签的相关性，容易出现0值
改进：
1. 使用分类准确率而非相关性
2. 分别用瞬态/持久特征训练简单分类器
3. 比较两者的判别能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def analyze_discriminative_transients_improved(model, dataloader, num_samples=100):
    """
    改进的瞬态特征判别性分析
    
    方法：
    1. 识别瞬态特征（lifetime < window_size）和持久特征（lifetime >= window_size）
    2. 用logistic回归评估两类特征的分类能力
    3. 比较AUC或准确率
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        num_samples: 分析样本数
        
    Returns:
        dict: 判别性分析结果
    """
    model.eval()
    device = model.device
    window_size = model.sae.window_size if model.use_sae else 8
    
    # 收集数据
    transient_feature_stats = []  # 每个样本的瞬态特征统计
    persistent_feature_stats = []  # 每个样本的持久特征统计
    labels = []
    
    print(f"收集瞬态/持久特征数据...")
    count = 0
    
    for batch in dataloader:
        if count >= num_samples:
            break
            
        inputs, batch_labels = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            _, interp = model(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
        
        for sample_idx in range(sparse_features.shape[0]):
            if count >= num_samples:
                break
            
            sample_features = sparse_features[sample_idx]  # [T, D]
            T, D = sample_features.shape
            active_mask = (sample_features > 0)  # [T, D]
            
            # 计算每个特征的激活持续时间
            transient_mask = torch.zeros(D, dtype=torch.bool, device=device)
            persistent_mask = torch.zeros(D, dtype=torch.bool, device=device)
            
            for feat_idx in range(D):
                feature_active = active_mask[:, feat_idx]
                
                if not feature_active.any():
                    continue
                
                # 计算连续激活的段落长度
                lifetimes = []
                current_lifetime = 0
                
                for t in range(T):
                    if feature_active[t]:
                        current_lifetime += 1
                    else:
                        if current_lifetime > 0:
                            lifetimes.append(current_lifetime)
                            current_lifetime = 0
                
                if current_lifetime > 0:
                    lifetimes.append(current_lifetime)
                
                if lifetimes:
                    median_lifetime = torch.tensor(lifetimes, dtype=torch.float32).median().item()
                    
                    if median_lifetime < window_size:
                        transient_mask[feat_idx] = True
                    else:
                        persistent_mask[feat_idx] = True
            
            # 提取特征统计量
            # 1. 平均激活强度
            # 2. 最大激活强度
            # 3. 激活频率（多少时间步被激活）
            # 4. 激活方差
            
            def compute_feature_stats(mask):
                if not mask.any():
                    return torch.zeros(4, device=device)
                
                selected_features = sample_features[:, mask]  # [T, num_selected]
                
                # 统计量
                mean_activation = selected_features.mean()
                max_activation = selected_features.max()
                activation_freq = (selected_features > 0).float().mean()
                activation_var = selected_features.var()
                
                return torch.tensor([
                    mean_activation,
                    max_activation,
                    activation_freq,
                    activation_var
                ], device=device)
            
            transient_stats = compute_feature_stats(transient_mask)
            persistent_stats = compute_feature_stats(persistent_mask)
            
            transient_feature_stats.append(transient_stats.cpu())
            persistent_feature_stats.append(persistent_stats.cpu())
            labels.append(batch_labels[sample_idx].cpu())
            
            count += 1
            
            if count % 20 == 0:
                print(f"  已处理 {count}/{num_samples} 样本")
    
    # 转换为tensor
    transient_features = torch.stack(transient_feature_stats)  # [N, 4]
    persistent_features = torch.stack(persistent_feature_stats)  # [N, 4]
    labels_tensor = torch.stack(labels)  # [N]
    
    print(f"\n瞬态特征统计: {transient_features.shape}")
    print(f"持久特征统计: {persistent_features.shape}")
    print(f"标签分布: bonafide={(labels_tensor==0).sum()}, spoof={(labels_tensor==1).sum()}")
    
    # 检查数据有效性
    if transient_features.abs().sum() == 0:
        print("⚠️ 警告: 没有检测到瞬态特征!")
        return {
            'transient_discriminative_power': 0.0,
            'persistent_discriminative_power': 0.0,
            'ratio': 0.0,
            'error': 'No transient features detected'
        }
    
    if persistent_features.abs().sum() == 0:
        print("⚠️ 警告: 没有检测到持久特征!")
        return {
            'transient_discriminative_power': 0.0,
            'persistent_discriminative_power': 0.0,
            'ratio': 0.0,
            'error': 'No persistent features detected'
        }
    
    # 方法1: Logistic回归分类能力
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import train_test_split
    
    # 分割训练/测试
    indices = torch.randperm(len(labels_tensor))
    split_idx = int(0.7 * len(indices))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    def evaluate_features(features, name):
        """评估特征的判别能力"""
        X_train = features[train_idx].numpy()
        y_train = labels_tensor[train_idx].numpy()
        X_test = features[test_idx].numpy()
        y_test = labels_tensor[test_idx].numpy()
        
        # 标准化
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        # 训练分类器
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        
        # 评估
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        
        # 只有在标签有两类时才计算AUC
        if len(set(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0.5
        
        print(f"\n{name}特征判别能力:")
        print(f"  准确率: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return acc, auc
    
    transient_acc, transient_auc = evaluate_features(transient_features, "瞬态")
    persistent_acc, persistent_auc = evaluate_features(persistent_features, "持久")
    
    # 方法2: 简单的相关性分析（改进版）
    def compute_correlation_safe(features, labels):
        """安全的相关性计算"""
        correlations = []
        
        for i in range(features.shape[1]):
            feat_col = features[:, i]
            
            # 检查方差
            if feat_col.std() < 1e-6:
                continue
            
            # Pearson相关系数
            feat_centered = feat_col - feat_col.mean()
            label_centered = labels.float() - labels.float().mean()
            
            corr = (feat_centered * label_centered).mean() / (feat_col.std() * labels.float().std() + 1e-8)
            correlations.append(abs(corr.item()))
        
        return torch.tensor(correlations).mean().item() if correlations else 0.0
    
    transient_corr = compute_correlation_safe(transient_features, labels_tensor)
    persistent_corr = compute_correlation_safe(persistent_features, labels_tensor)
    
    print(f"\n相关性分析:")
    print(f"  瞬态特征相关性: {transient_corr:.4f}")
    print(f"  持久特征相关性: {persistent_corr:.4f}")
    
    # 综合结果（使用AUC作为主要指标）
    transient_power = transient_auc
    persistent_power = persistent_auc
    ratio = transient_power / (persistent_power + 1e-8)
    
    print(f"\n" + "="*60)
    print(f"瞬态分析结果:")
    print(f"  瞬态特征判别力: {transient_power:.4f}")
    print(f"  持久特征判别力: {persistent_power:.4f}")
    print(f"  比值 (瞬态/持久): {ratio:.4f}")
    
    if ratio > 0.8:
        print(f"  ⚠️ 瞬态特征具有较强判别能力，window可能过度平滑了重要信号!")
    elif ratio > 0.5:
        print(f"  ℹ️ 瞬态特征有一定判别能力")
    else:
        print(f"  ✓ 持久特征更重要，window平滑是合理的")
    print("="*60)
    
    return {
        'transient_discriminative_power': transient_power,
        'persistent_discriminative_power': persistent_power,
        'ratio': ratio,
        'transient_accuracy': transient_acc,
        'persistent_accuracy': persistent_acc,
        'transient_correlation': transient_corr,
        'persistent_correlation': persistent_corr,
        'num_samples': count,
        'num_transient_features': (transient_features.abs().sum(dim=1) > 0).sum().item(),
        'num_persistent_features': (persistent_features.abs().sum(dim=1) > 0).sum().item(),
    }


if __name__ == '__main__':
    # 测试代码
    print("使用说明：")
    print("1. 在model_window_topk.py中替换analyze_discriminative_transients方法")
    print("2. 或者直接导入此函数使用")
    print("\n示例:")
    print("from improved_transient_analysis import analyze_discriminative_transients_improved")
    print("results = analyze_discriminative_transients_improved(model, dataloader, num_samples=100)")
