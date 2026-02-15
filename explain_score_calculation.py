#!/usr/bin/env python
"""
演示 discrimination score 的计算过程
"""

import torch
import numpy as np

print("=" * 70)
print("Discrimination Score 计算演示")
print("=" * 70)

# 模拟数据：假设我们有5个样本，3个features（实际是250样本，4096 features）
print("\n【步骤1】收集样本的feature激活值")
print("-" * 70)

# Bonafide样本的激活值 [num_bonafide_samples, num_features]
bonafide_features = torch.tensor([
    [0.8, 0.2, 0.1],  # bonafide样本1的3个features激活值
    [0.7, 0.3, 0.0],  # bonafide样本2
    [0.9, 0.1, 0.2],  # bonafide样本3
    [0.6, 0.4, 0.1],  # bonafide样本4
    [0.8, 0.2, 0.0],  # bonafide样本5
])

# Spoof样本的激活值 [num_spoof_samples, num_features]
spoof_features = torch.tensor([
    [0.1, 0.9, 0.3],  # spoof样本1的3个features激活值
    [0.2, 0.8, 0.2],  # spoof样本2
    [0.0, 1.0, 0.4],  # spoof样本3
    [0.3, 0.7, 0.3],  # spoof样本4
    [0.1, 0.9, 0.2],  # spoof样本5
])

print(f"Bonafide样本: {bonafide_features.shape[0]}个")
print(f"Spoof样本: {spoof_features.shape[0]}个")
print(f"每个样本的feature维度: {bonafide_features.shape[1]}")

print(f"\nBonafide激活值矩阵:")
print(bonafide_features.numpy())
print(f"\nSpoof激活值矩阵:")
print(spoof_features.numpy())

print("\n【步骤2】计算每个feature的平均激活值")
print("-" * 70)

# 对每个feature，在所有bonafide样本上取平均
bonafide_mean = bonafide_features.mean(dim=0)  # [num_features]
print(f"Bonafide平均激活值 (对每个feature): {bonafide_mean.numpy()}")
print(f"  Feature 0: {bonafide_mean[0]:.4f} (在5个bonafide样本上的平均)")
print(f"  Feature 1: {bonafide_mean[1]:.4f}")
print(f"  Feature 2: {bonafide_mean[2]:.4f}")

# 对每个feature，在所有spoof样本上取平均
spoof_mean = spoof_features.mean(dim=0)  # [num_features]
print(f"\nSpoof平均激活值 (对每个feature): {spoof_mean.numpy()}")
print(f"  Feature 0: {spoof_mean[0]:.4f} (在5个spoof样本上的平均)")
print(f"  Feature 1: {spoof_mean[1]:.4f}")
print(f"  Feature 2: {spoof_mean[2]:.4f}")

print("\n【步骤3】计算discrimination score")
print("-" * 70)
print("公式: score = |bonafide_mean - spoof_mean|")
print("对4096个features中的每一个都计算这个score\n")

# 计算每个feature的discrimination score
diff = torch.abs(bonafide_mean - spoof_mean)
print(f"Discrimination scores: {diff.numpy()}")
print(f"\n详细计算:")
for i in range(len(diff)):
    print(f"  Feature {i}: |{bonafide_mean[i]:.4f} - {spoof_mean[i]:.4f}| = {diff[i]:.4f}")

print("\n【步骤4】找出最重要的features")
print("-" * 70)

# 按score排序，找出top features
top_k = 3
top_scores, top_indices = diff.topk(top_k)
print(f"Top {top_k} 最重要的features:")
for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
    print(f"  {rank}. Feature {idx.item()}: score={score.item():.4f}")
    print(f"     → Bonafide平均={bonafide_mean[idx]:.4f}, Spoof平均={spoof_mean[idx]:.4f}")
    
    if bonafide_mean[idx] > spoof_mean[idx]:
        print(f"     → 这是bonafide-specific feature (在真语音中激活更强)")
    else:
        print(f"     → 这是spoof-specific feature (在假语音中激活更强)")

print("\n" + "=" * 70)
print("实际分析中的规模")
print("=" * 70)
print("样本数量: 250 bonafide + 250 spoof")
print("Feature维度: 4096 (TopK SAE的字典大小)")
print("计算: 对4096个features，每个都计算 |bonafide_mean - spoof_mean|")
print("\n结果示例:")
print("  Feature 1626: score=0.1036")
print("    意思是: Feature 1626在bonafide和spoof上的平均激活值相差0.1036")
print("    这个差异越大，说明这个feature对区分真假语音越重要")
