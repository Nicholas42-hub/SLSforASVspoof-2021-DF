#!/usr/bin/env python
"""
演示Feature Discrimination Score的实际计算过程
"""

import torch
import numpy as np

print("="*70)
print("Feature Discrimination Score 计算演示")
print("="*70)

# 模拟实际数据
print("\n1. 模拟样本数据")
print("-" * 70)

# 假设我们有以下样本
num_bonafide = 49
num_spoof = 463
num_features = 4096

# 创建模拟的feature activations
# 在实际模型中，这些来自SAE的sparse encoding
np.random.seed(42)
bonafide_features = np.random.rand(num_bonafide, num_features) * 0.5
spoof_features = np.random.rand(num_spoof, num_features) * 0.5

# 模拟TopK sparsity: 大部分值设为0
for i in range(num_bonafide):
    mask = np.random.choice(num_features, size=num_features-128, replace=False)
    bonafide_features[i, mask] = 0

for i in range(num_spoof):
    mask = np.random.choice(num_features, size=num_features-128, replace=False)
    spoof_features[i, mask] = 0

print(f"Bonafide samples: {num_bonafide}, shape: {bonafide_features.shape}")
print(f"Spoof samples: {num_spoof}, shape: {spoof_features.shape}")

# 2. 计算每个feature的平均激活
print("\n2. 计算平均激活值")
print("-" * 70)

bonafide_mean = bonafide_features.mean(axis=0)  # [4096]
spoof_mean = spoof_features.mean(axis=0)        # [4096]

print(f"bonafide_mean shape: {bonafide_mean.shape}")
print(f"spoof_mean shape: {spoof_mean.shape}")

# 3. 计算discrimination score
print("\n3. 计算Discrimination Score")
print("-" * 70)

diff = np.abs(bonafide_mean - spoof_mean)
print(f"差异向量 shape: {diff.shape}")
print(f"差异范围: [{diff.min():.6f}, {diff.max():.6f}]")

# 4. 找出Top discriminative features
print("\n4. Top 20 最具判别性的Features")
print("-" * 70)

# 获取Top 20
top_indices = np.argsort(diff)[::-1][:20]
top_scores = diff[top_indices]

for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
    bon_val = bonafide_mean[idx]
    spo_val = spoof_mean[idx]
    preference = "Bonafide" if bon_val > spo_val else "Spoof"
    
    print(f"{rank:3d}. Feature {idx:4d}: score={score:.6f} "
          f"(bonafide={bon_val:.4f}, spoof={spo_val:.4f}, prefer={preference})")

# 5. 详细分析Top 1 feature
print("\n5. 详细分析Feature", top_indices[0])
print("-" * 70)

feature_id = top_indices[0]

bon_activations = bonafide_features[:, feature_id]
spo_activations = spoof_features[:, feature_id]

print(f"Feature {feature_id} 在Bonafide样本上:")
print(f"  非零激活: {(bon_activations > 0).sum()}/{num_bonafide}")
print(f"  平均值: {bon_activations.mean():.6f}")
print(f"  标准差: {bon_activations.std():.6f}")
print(f"  范围: [{bon_activations.min():.6f}, {bon_activations.max():.6f}]")

print(f"\nFeature {feature_id} 在Spoof样本上:")
print(f"  非零激活: {(spo_activations > 0).sum()}/{num_spoof}")
print(f"  平均值: {spo_activations.mean():.6f}")
print(f"  标准差: {spo_activations.std():.6f}")
print(f"  范围: [{spo_activations.min():.6f}, {spo_activations.max():.6f}]")

print(f"\nDiscrimination Score = |{bon_activations.mean():.6f} - {spo_activations.mean():.6f}| = {top_scores[0]:.6f}")

if bon_activations.mean() > spo_activations.mean():
    print(f"→ 这个feature更倾向于Bonafide语音")
    print(f"  可能捕捉到真实语音的特征（如自然韵律、呼吸声等）")
else:
    print(f"→ 这个feature更倾向于Spoof语音")
    print(f"  可能捕捉到伪造语音的特征（如TTS机械感、VC artifacts等）")

# 6. 统计总体判别力
print("\n6. 总体判别力统计")
print("-" * 70)

print(f"Top 50 features平均discrimination score: {diff[top_indices[:50]].mean():.6f}")
print(f"所有features平均discrimination score: {diff.mean():.6f}")
print(f"最大discrimination score: {diff.max():.6f}")
print(f"最小discrimination score: {diff.min():.6f}")

# 判别力评级
avg_top50 = diff[top_indices[:50]].mean()
if avg_top50 > 0.1:
    rating = "强判别力 ✓"
elif avg_top50 > 0.05:
    rating = "中等判别力 ~"
else:
    rating = "弱判别力 ✗"

print(f"\n判别力评级: {rating}")

# 7. 类别特异性分析
print("\n7. 类别特异性Features")
print("-" * 70)

# Bonafide-specific: bonafide激活 > spoof激活的3倍
bonafide_specific = (bonafide_mean > spoof_mean * 3) & ((bonafide_features > 0).mean(axis=0) > 0.3)
spoof_specific = (spoof_mean > bonafide_mean * 3) & ((spoof_features > 0).mean(axis=0) > 0.3)

print(f"Bonafide-specific neurons: {bonafide_specific.sum()}")
print(f"Spoof-specific neurons: {spoof_specific.sum()}")

if bonafide_specific.sum() > 0:
    print(f"\nBonafide-specific feature示例:")
    bon_spec_idx = np.where(bonafide_specific)[0][:5]
    for idx in bon_spec_idx:
        print(f"  Feature {idx}: bonafide={bonafide_mean[idx]:.4f}, spoof={spoof_mean[idx]:.4f}, "
              f"ratio={bonafide_mean[idx]/max(spoof_mean[idx], 0.001):.2f}x")

if spoof_specific.sum() > 0:
    print(f"\nSpoof-specific feature示例:")
    spo_spec_idx = np.where(spoof_specific)[0][:5]
    for idx in spo_spec_idx:
        print(f"  Feature {idx}: bonafide={bonafide_mean[idx]:.4f}, spoof={spoof_mean[idx]:.4f}, "
              f"ratio={spoof_mean[idx]/max(bonafide_mean[idx], 0.001):.2f}x")

print("\n" + "="*70)
print("计算完成！")
print("="*70)
print("\n关键要点:")
print("• Discrimination Score = |bonafide平均 - spoof平均|")
print("• Score越高，该feature的判别能力越强")
print("• Top features是模型做决策时的关键依据")
print("• 你的实际结果中，Top score约0.08，属于中等水平")
