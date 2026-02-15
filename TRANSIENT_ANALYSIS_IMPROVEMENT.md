# 瞬态分析改进说明

## 🔧 改进内容

### 原始方法的问题
```python
# 问题1: 只用单一统计量（平均值）
transient_activations = sample_features[:, transient_features].mean()

# 问题2: 使用简单的相关系数
correlation = cov / (x_std * y_std)  # 容易出现数值不稳定

# 问题3: 没有错误处理
# 结果: 全是 0.0
```

### 改进方法

#### 1. **多维特征表示** (4维)
```python
transient_stats = [
    mean_activation,     # 平均激活强度
    max_activation,      # 最大激活强度
    activation_freq,     # 激活频率（时间步比例）
    activation_var       # 激活方差
]
```

#### 2. **使用 Logistic Regression + AUC**
```python
# 训练简单分类器评估判别能力
clf = LogisticRegression()
clf.fit(transient_features, labels)

# 使用 AUC 作为判别能力指标
transient_auc = roc_auc_score(y_test, y_prob)
persistent_auc = roc_auc_score(y_test, y_prob)

ratio = transient_auc / persistent_auc
```

**优势:**
- AUC 在 [0.5, 1.0] 范围，更稳定
- 不会出现 0 值（除非完全随机）
- 对类别不平衡更鲁棒

#### 3. **数据验证**
```python
# 检查特征是否为空
if transient_features.abs().sum() == 0:
    return {'error': 'No transient features detected'}

# 检查标签分布
print(f"bonafide={...}, spoof={...}")

# 标准化特征
X = (X - mean) / std
```

#### 4. **Fallback 机制**
```python
try:
    # 优先使用 sklearn
    from sklearn.linear_model import LogisticRegression
    # ... AUC 评估
except ImportError:
    # 降级到改进的相关性方法
    # ... 多特征相关性平均
```

---

## 📊 新增输出指标

```json
{
  "transient_discriminative_power": 0.65,      // AUC (0.5-1.0)
  "persistent_discriminative_power": 0.78,     // AUC (0.5-1.0)
  "ratio": 0.83,                               // transient/persistent
  "transient_accuracy": 0.62,                  // 分类准确率
  "persistent_accuracy": 0.74,
  "num_samples": 100,
  "num_transient_samples": 95,                 // 有瞬态特征的样本数
  "num_persistent_samples": 98                 // 有持久特征的样本数
}
```

---

## 🎯 解读指南

### Ratio 阈值

| Ratio | 解释 | 建议 |
|-------|------|------|
| > 0.8 | 瞬态特征高度判别 | 🚨 Window 严重损害性能，考虑去除 |
| 0.5-0.8 | 瞬态特征显著判别 | ⚠️ Window 可能次优，考虑减小 window_size |
| 0.3-0.5 | 瞬态特征中等判别 | ℹ️ 需权衡 stability vs performance |
| < 0.3 | 持久特征占主导 | ✅ Window smoothing 合理 |

### 示例场景

**场景 A: Window 是好的**
```json
{
  "transient_auc": 0.55,
  "persistent_auc": 0.82,
  "ratio": 0.67
}
```
→ 持久特征更重要，window constraint 合理

**场景 B: Window 有问题**
```json
{
  "transient_auc": 0.78,
  "persistent_auc": 0.68,
  "ratio": 1.15
}
```
→ 瞬态特征更判别！Window 可能平滑掉了 spoofing artifacts

---

## 🚀 使用方法

### 运行改进的分析
```bash
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF
mkdir -p logs
sbatch rerun_limitations_analysis.slurm
```

### 查看结果
```bash
# 查看新的分析结果
cat window_limitations_analysis_v2/limitations_analysis.json

# 对比旧结果
diff window_limitations_analysis/limitations_analysis.json \
     window_limitations_analysis_v2/limitations_analysis.json
```

---

## 📁 修改的文件

1. **model_window_topk.py**
   - 替换 `analyze_discriminative_transients()` 方法
   - 新增多维特征统计
   - 集成 sklearn Logistic Regression

2. **analyze_window_limitations.py**
   - 更新输出信息显示 AUC
   - 添加更详细的判断阈值
   - 改进错误处理

3. **rerun_limitations_analysis.slurm**
   - 新增重新运行脚本

---

## ✅ 预期改进

| 指标 | 旧方法 | 新方法 |
|------|--------|--------|
| transient_power | 0.0 ❌ | 0.55-0.85 ✅ |
| persistent_power | 0.0 ❌ | 0.60-0.90 ✅ |
| ratio | 0.0 ❌ | 0.5-1.2 ✅ |
| 可解释性 | ❓ | 明确 ✅ |

现在应该能得到有意义的数值了！
