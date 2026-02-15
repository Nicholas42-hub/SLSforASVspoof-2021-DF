# 深度分析实验指南

## 三个分析任务

### 1. 可视化Top-10决策特征的激活模式 
**目的**: 理解决策特征捕捉什么声学pattern，为什么更稳定

**脚本**: `visualize_decision_features.py`

**功能**:
- 提取top-10决策相关特征（从decision_analysis结果中）
- 对genuine和spoof样本分别可视化
- 生成三层可视化：
  * Mel频谱图（底层声学）
  * 决策特征激活热图（10个特征 × 时间）
  * 单个特征的激活轨迹
- 分析统计特性：mean, std, sparsity, temporal consistency

**运行**:
```bash
sbatch run_feature_visualization.slurm
```

**输出**:
- `decision_feature_visualizations/` 目录
- 每个样本的可视化图: `{sample_name}_activation_pattern.png`
- 终端输出: 每个特征的统计分析

**预期发现**:
- 决策特征可能对应特定的声学事件（如phoneme boundaries, attack artifacts）
- 高稳定性可能来自于对持续性声学pattern的响应
- Genuine vs Spoof的激活模式差异

---

### 2. 边界语义分析
**目的**: 检查boundary discontinuities是噪音还是有意义的信号

**脚本**: `analyze_boundary_semantics.py`

**功能**:
- 检测所有window边界位置
- 对每个样本计算:
  * Boundary frames的Jaccard相似度
  * Interior frames的Jaccard相似度
  * 是否预测正确
  * 预测置信度
- 统计分析: 正确vs错误预测的boundary discontinuity差异
- 统计检验: t-test, effect size (Cohen's d)

**运行**:
```bash
sbatch run_boundary_analysis.slurm
```

**输出**:
- `boundary_semantic_analysis/` 目录
- `boundary_error_correlation.png`: 4个子图
  * Boundary discontinuity分布（正确vs错误）
  * Interior discontinuity分布
  * Discontinuity vs 置信度散点图
  * 箱线图对比
- `boundary_analysis_results.json`: 详细数据
- 终端输出: 统计检验结果

**预期发现**:
- 如果boundary jumps和错误**无相关性**: 说明是architecture artifact，应该修复
- 如果boundary jumps和错误**正相关**: 说明不稳定会导致错误决策
- 如果boundary jumps和错误**负相关**: 说明可能是检测attack transition的信号！

---

### 3. 与人工特征对比
**目的**: 验证SAE学到的特征是否比MFCC等手工特征更有意义

**脚本**: `compare_handcrafted_features.py`

**功能**:
- 提取三种特征表示:
  * SAE learned features (4096维稀疏)
  * MFCC (40维)
  * Log Mel-Spectrogram (80维)
- 对每种特征计算temporal stability:
  * Cosine similarity (frame-to-frame)
  * Feature lifetime (持续帧数)
  * Jaccard similarity (仅SAE，因为稀疏)
- 在500个样本上统计对比

**运行**:
```bash
sbatch run_feature_comparison.slurm
```

**输出**:
- `feature_comparison_analysis/` 目录
- `sae_vs_handcrafted_comparison.png`: 对比可视化
  * 左图: Cosine similarity箱线图
  * 右图: Feature lifetime箱线图
- `comparison_results.json`: 汇总统计
- 终端输出: 详细对比和解释

**预期发现**:
- 如果SAE更稳定: 说明学到了更高层次的语义特征
- 如果MFCC更稳定: 说明底层声学特征本身就很平滑，SAE的稀疏性引入了变化
- Feature lifetime对比: 揭示SAE是否捕捉更长时间尺度的pattern

---

## 批量运行所有分析

```bash
# 提交所有三个任务
sbatch run_feature_visualization.slurm
sbatch run_boundary_analysis.slurm  
sbatch run_feature_comparison.slurm

# 监控任务
squeue -u $USER

# 查看输出
tail -f slurm-*.out
```

---

## 预期时间

| 任务 | 样本数 | 预计时间 |
|------|--------|----------|
| 可视化 | 10 (5+5) | ~20分钟 |
| 边界分析 | 2000 | ~1.5小时 |
| 特征对比 | 500 | ~30分钟 |

**总计**: 约2-3小时

---

## 结果整合

分析完成后，你将有：

1. **可视化证据**: Top-10特征到底在检测什么
2. **统计证据**: Boundary jumps是否影响性能
3. **对比证据**: SAE vs 传统特征的优劣

这些将支撑你的核心论点：
> "Window TopK让决策特征自然稳定，且学到的表示比手工特征更有意义"

---

## 下一步（分析完成后）

基于结果，你可以：

### 如果发现SAE特征显著更好:
→ 写paper主打"learned representations for interpretable detection"

### 如果发现boundary jumps有害:
→ 实现targeted boundary smoothing作为minor contribution

### 如果发现boundary jumps有信息:
→ 重新定位: boundary discontinuity是feature而非bug!

---

## 调试Tips

如果遇到错误：

1. **OOM (Out of Memory)**:
   - 减少批量大小
   - 减少样本数量

2. **模型加载失败**:
   ```python
   # 检查模型路径
   ls -lh models/model_window_topk_k128_continue/
   ```

3. **数据路径错误**:
   ```python
   # 验证数据集路径
   ls /data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval/
   ```

4. **可视化不显示**:
   - 脚本已经设置为保存到文件而非显示
   - 检查输出目录是否创建成功

---

Good luck! 🚀
