# 错误分类可视化指南 (Incorrect Classification Visualization Guide)

## 概述 (Overview)

新增功能：专门针对**错误分类样本**的注意力热图可视化，帮助分析模型为什么会出错。

## 两种错误类型 (Two Types of Errors)

### 1. False Rejection (假拒绝)

- **真实标签**: Bonafide (真实语音)
- **模型预测**: Spoof (伪造语音)
- **含义**: 模型错误地将真实语音判断为伪造语音
- **可视化文件**: `layer_weight_incorrect_bonafide.png`

### 2. False Acceptance (假接受)

- **真实标签**: Spoof (伪造语音)
- **模型预测**: Bonafide (真实语音)
- **含义**: 模型错误地将伪造语音判断为真实语音
- **可视化文件**: `layer_weight_incorrect_spoof.png`

## 使用方法 (Usage)

### 方法 1: 仅生成错误分类可视化

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /path/to/best_model_eer_g3_viz_only.pth \
  --database_path /path/to/ASVspoof2021_LA_eval/ \
  --protocols_path /path/to/trial_metadata.txt \
  --track LA \
  --viz_dir attention_viz_incorrect_only \
  --num_viz_samples 500 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --incorrect_only_viz
```

**关键参数**:

- `--has_labels`: 必须设置，表示数据集有真实标签
- `--incorrect_only_viz`: 新参数，仅生成错误分类可视化
- `--num_viz_samples`: 建议设置较大值（如 500），确保收集到足够的错误样本

### 方法 2: 生成所有可视化（包括错误分类）

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /path/to/best_model_eer_g3_viz_only.pth \
  --database_path /path/to/ASVspoof2021_LA_eval/ \
  --protocols_path /path/to/trial_metadata.txt \
  --track LA \
  --viz_dir attention_viz_complete \
  --num_viz_samples 500 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --classification_viz \
  --incorrect_only_viz
```

## 生成的可视化文件 (Generated Visualizations)

使用 `--incorrect_only_viz` 会生成以下文件：

### 1. 标准可视化 (Standard Visualizations)

- `layer_weight_heatmap_spoof.png` - 所有 spoof 样本的层级注意力
- `layer_weight_heatmap_bonafide.png` - 所有 bonafide 样本的层级注意力
- `layer_weight_comparison.png` - 双图对比
- `temporal_attention_heatmap_*.png` - 时间维度注意力
- `intra_group_attention_*.png` - 组内注意力
- `inter_group_attention_*.png` - 组间注意力

### 2. 错误分类专用可视化 (Incorrect-Only Visualizations)

- **`layer_weight_incorrect_bonafide.png`**

  - 显示：被错误分类为 spoof 的 bonafide 样本（False Rejection）
  - 样本数：10 个（可调整）
  - 分析：这些真实语音有什么特征导致模型误判？

- **`layer_weight_incorrect_spoof.png`**

  - 显示：被错误分类为 bonafide 的 spoof 样本（False Acceptance）
  - 样本数：10 个（可调整）
  - 分析：这些伪造语音有什么特征欺骗了模型？

- **`incorrect_classification_comparison.png`**
  - 并排对比图（1×2 布局）
  - 左图：False Rejection (Bonafide → Spoof)
  - 右图：False Acceptance (Spoof → Bonafide)
  - 便于直接对比两种错误类型的注意力模式差异

## 输出示例 (Output Example)

```
❌ GENERATING INCORRECT CLASSIFICATION VISUALIZATIONS
======================================================================

📊 Error Analysis:
   False Rejections (Bonafide → Spoof): 45 samples
   False Acceptances (Spoof → Bonafide): 23 samples

📊 Generating individual error heatmaps...
💾 Saved Incorrectly Classified Bonafide (False Rejection) heatmap: attention_viz_incorrect_only/layer_weight_incorrect_bonafide.png
💾 Saved Incorrectly Classified Spoof (False Acceptance) heatmap: attention_viz_incorrect_only/layer_weight_incorrect_spoof.png

🔍 Generating incorrect classification comparison...
💾 Saved incorrect classification comparison: attention_viz_incorrect_only/incorrect_classification_comparison.png

======================================================================
✅ INCORRECT CLASSIFICATION VISUALIZATIONS SAVED TO: attention_viz_incorrect_only
======================================================================
```

## 分析建议 (Analysis Tips)

### 比较正确与错误分类

1. 先运行 `--classification_viz` 获取所有 4 类可视化（正确 bonafide、错误 bonafide、正确 spoof、错误 spoof）
2. 对比：
   - Correct bonafide vs Incorrect bonafide → 什么导致了 False Rejection？
   - Correct spoof vs Incorrect spoof → 什么导致了 False Acceptance？

### 关注层级注意力模式

- **False Rejection**: 检查哪些层的注意力分布与正确分类的 bonafide 样本不同
- **False Acceptance**: 检查哪些层的注意力分布与正确分类的 spoof 样本不同

### 样本数量调整

如果某类错误样本较少，可以在代码中调整：

```python
# 在 evaluate_with_attention_viz.py 中修改
visualizer.generate_incorrect_only_visualizations(num_samples_per_category=5)
# 改为更小的值，例如 num_samples_per_category=3
```

## 服务器运行示例 (Server Example)

```bash
# SSH到服务器
ssh root@your-server

# 进入项目目录
cd /root/autodl-tmp/SLSforASVspoof-2021-DF

# 运行错误分类可视化
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint models/g3_heatmap_LA_CCE_100_16_1e-06_group3_contrastiveFalse_g3_viz_only/best_model_eer_g3_viz_only.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/trial_metadata.txt \
  --track LA \
  --viz_dir attention_viz_LA_incorrect_analysis \
  --num_viz_samples 500 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --incorrect_only_viz
```

## 技术细节 (Technical Details)

### 样本选择逻辑

- 从所有收集的样本中筛选：`true_label != prediction`
- False Rejection: `true_label=1 (bonafide) && prediction=0 (spoof)`
- False Acceptance: `true_label=0 (spoof) && prediction=1 (bonafide)`

### 注意力权重计算

- 时间维度注意力 `(L, T)` 平均为层级重要性 `(L,)`
- 每个样本显示 24 层（或您的模型层数）的注意力分布

### 可视化配置

- 颜色映射: `viridis`
- 分辨率: 300 DPI
- 格式: PNG
- 热图尺寸: 1×2 对比图为 20×8 英寸

## 常见问题 (FAQ)

**Q: 如果没有错误分类样本怎么办？**
A: 系统会显示警告并跳过可视化：

```
⚠️  Warning: All predictions are correct (100% accuracy).
   No incorrectly classified samples to visualize.
```

**Q: 可以只生成 False Rejection 或只生成 False Acceptance 吗？**
A: 可以，修改 `generate_incorrect_only_visualizations` 函数，注释掉不需要的部分。

**Q: 错误样本数量不足 10 个怎么办？**
A: 系统会自动调整，使用所有可用样本并显示警告。

## 相关文件 (Related Files)

- `evaluate_with_attention_viz.py` - 主评估脚本
- `visualize_attention_evaluation.py` - 可视化核心代码
  - `generate_incorrect_only_visualizations()` - 新增函数
  - `plot_incorrect_comparison()` - 新增函数
- `CLASSIFICATION_VISUALIZATION_GUIDE.md` - 完整分类可视化指南
