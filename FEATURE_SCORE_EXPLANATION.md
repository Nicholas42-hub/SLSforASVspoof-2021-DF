# Feature Discrimination Score 计算方法详解

## 核心计算公式

```python
# 1. 收集所有样本的feature activation
bonafide_features = []  # 收集所有bonafide样本的sparse features
spoof_features = []     # 收集所有spoof样本的sparse features

# 2. 计算每个feature的平均激活值
bonafide_mean = bonafide_features.mean(dim=0)  # [4096]
spoof_mean = spoof_features.mean(dim=0)        # [4096]

# 3. 计算discrimination score
diff = torch.abs(bonafide_mean - spoof_mean)   # [4096]

# 4. 选择Top 50
top_discriminative = diff.topk(50)
# top_discriminative.indices = [1626, 3466, 2739, ...]  feature编号
# top_discriminative.values = [0.083, 0.080, 0.078, ...]  判别分数
```

## 具体例子

### Feature 1626 (score=0.083)

假设在分析的500个样本中（bonafide=49, spoof=463）:

```python
# Feature 1626在所有bonafide样本上的平均激活值
bonafide_mean[1626] = 0.245  # 比如平均激活强度为0.245

# Feature 1626在所有spoof样本上的平均激活值  
spoof_mean[1626] = 0.162     # 比如平均激活强度为0.162

# Discrimination score
score = |0.245 - 0.162| = 0.083
```

**解释**: Feature 1626在bonafide样本上的平均激活比在spoof样本上高0.083，说明这个feature更倾向于bonafide语音的特征。

### Feature 3466 (score=0.080)

```python
# 可能的情况1: Spoof-specific feature
bonafide_mean[3466] = 0.120
spoof_mean[3466] = 0.200
score = |0.120 - 0.200| = 0.080
```

**解释**: Feature 3466在spoof样本上激活更强，可能捕捉到伪造语音的特有特征（如TTS的机械感、VC的artifacts）。

## 计算流程详解

### Step 1: 特征收集

对于每个样本，模型forward返回的`interp['avg_activation']`是一个4096维向量:

```python
# 单个样本的sparse features
sample_features = [0, 0.5, 0, 0.8, 0, ..., 0.3, 0]  # 4096维
                   ↑   ↑                     ↑
              未激活  激活                  激活
```

TopK SAE确保每个样本只有k=128个features被激活（非零）。

### Step 2: 分类汇总

```python
# 按标签分组
if label == bonafide:
    bonafide_features.append(sample_features)
else:
    spoof_features.append(sample_features)

# 最终得到
bonafide_features: [49, 4096]  # 49个bonafide样本
spoof_features:    [463, 4096] # 463个spoof样本
```

### Step 3: 计算均值

```python
# 对每个feature维度求平均
bonafide_mean = bonafide_features.mean(dim=0)  # [4096]

# 例如Feature 1626:
# bonafide_mean[1626] = (sample1[1626] + sample2[1626] + ... + sample49[1626]) / 49
```

### Step 4: 计算差异

```python
# 对4096个features逐个计算
for i in range(4096):
    diff[i] = |bonafide_mean[i] - spoof_mean[i]|

# diff[1626] = 0.083  表示Feature 1626在两类样本上的激活差异最大
```

## Score的物理意义

### 高Score (>0.1)
- **强判别力**: 这个feature在bonafide和spoof上的激活模式差异很大
- **可靠指标**: 可以作为分类的重要依据
- **可解释性**: 容易识别和理解

### 中等Score (0.05-0.1)
- **中等判别力**: 有一定的区分能力，但不够显著
- **辅助作用**: 需要结合其他features判断
- **当前情况**: 你的模型大部分features在这个范围

### 低Score (<0.05)
- **弱判别力**: 两类样本上激活模式相似
- **噪声特征**: 可能只是随机变化
- **可忽略**: 对分类贡献小

## 数值示例

### 理想情况（强判别力）

```python
Feature X: 
  bonafide samples: [0.8, 0.9, 0.7, 0.85, ...]  → mean = 0.80
  spoof samples:    [0.1, 0.0, 0.2, 0.15, ...]  → mean = 0.15
  score = |0.80 - 0.15| = 0.65  ← 非常高的判别力！
```

### 你的模型情况（中等判别力）

```python
Feature 1626:
  bonafide samples: [0.3, 0.2, 0.25, 0.35, ...]  → mean ≈ 0.28
  spoof samples:    [0.2, 0.19, 0.21, 0.18, ...]  → mean ≈ 0.20
  score = |0.28 - 0.20| = 0.08  ← 中等判别力
```

### 弱判别特征

```python
Feature Y:
  bonafide samples: [0.15, 0.18, 0.16, ...]  → mean = 0.16
  spoof samples:    [0.17, 0.15, 0.16, ...]  → mean = 0.16
  score = |0.16 - 0.16| = 0.00  ← 无判别力
```

## 为什么你的Score相对较低？

### 可能原因

1. **样本不平衡**
   - Bonafide: 49个样本
   - Spoof: 463个样本
   - 比例 ≈ 1:9.4，可能导致统计不稳定

2. **训练尚未完全收敛**
   - 当前epoch 11/40 (27.5%)
   - SAE还在学习阶段，features可能还不够discriminative

3. **SAE权重较小**
   - sae_weight = 0.1
   - reconstruction loss权重相对分类loss较小
   - 可能导致SAE更关注重建而非判别

4. **k值设置**
   - k=128, sparsity=3.12%
   - 但实际观察到12.7%的sparsity，说明激活比预期密集
   - 可能需要调整k或SAE架构

## 如何提高Discrimination Score

### 方法1: 增加SAE权重
```bash
# 当前: sae_weight=0.1
# 尝试: sae_weight=0.2, 0.3, 0.5
```

### 方法2: 调整k值
```bash
# 当前: k=128 (期望3.12%, 实际12.7%)
# 尝试: k=64 (更稀疏) 或 k=256 (更密集)
```

### 方法3: 平衡数据采样
```python
# 对bonafide和spoof均匀采样
bonafide_samples = 500
spoof_samples = 500
```

### 方法4: 继续训练
等待模型训练到更多epochs，SAE有机会学习更discriminative的features。

## 实际应用

### 查看Top features的激活模式

已生成的可视化`feature_discrimination.png`展示了:
- **绿色条**: bonafide samples上的平均激活
- **红色条**: spoof samples上的平均激活
- **条高差异**: 对应discrimination score

### 识别Bonafide vs Spoof特征

```python
# Bonafide-specific: bonafide激活 >> spoof激活
if bonafide_mean[i] > spoof_mean[i] * 3:
    # 这个feature倾向于真实语音
    
# Spoof-specific: spoof激活 >> bonafide激活  
if spoof_mean[i] > bonafide_mean[i] * 3:
    # 这个feature倾向于伪造语音
```

当前模型找到了:
- 8个bonafide-specific features
- 8个spoof-specific features

## 总结

**Discrimination Score = |bonafide平均激活 - spoof平均激活|**

- 衡量一个feature在两类样本上的激活差异
- Score越高，该feature的判别力越强
- 当前模型score偏低(0.05-0.08)，但在可接受范围
- 随着训练进行，score应该会提高
- 可通过调整hyperparameters进一步优化
