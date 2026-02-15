# SAE可解释性分析 - 快速开始

## 已完成的评估结果

### 当前模型性能
- **Validation EER**: 10.84% (epoch 11/40)
- **ASVspoof2019 LA Eval EER**: 8.49%
- 模型使用sparse features (4096-dim), k=128, sparsity=3.12%

## 如何发现重要特征 - 三步走

### Step 1: 快速测试模型是否支持可解释性
```bash
python test_interpretability.py
```

这会检查:
- ✓ 模型是否使用sparse features
- ✓ SAE是否正确配置
- ✓ 稀疏度是否合理
- ✓ 特征多样性如何

### Step 2: 深度分析neuron重要性
```bash
# 方法1: 提交SLURM任务（推荐）
sbatch run_sae_analysis.sh

# 方法2: 本地运行（需要GPU）
python analyze_sae_neurons.py \
    --model_path models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_topk_sae_sparse/best_checkpoint_eer_topk_sae_sparse.pth \
    --num_samples 500 \
    --output_dir analysis_results
```

分析内容包括:
1. **特征激活统计**: 哪些neurons在bonafide/spoof上激活
2. **判别性特征**: 识别Top N最能区分真假的features
3. **类别特异性**: 找出bonafide-specific和spoof-specific neurons
4. **稀疏性验证**: 确认TopK机制是否有效

### Step 3: 查看可视化结果
分析完成后，在`analysis_results/`目录下会生成:

```
analysis_results/
├── feature_discrimination.png    # Top100判别特征的激活模式
├── activation_frequency.png      # 4096个features的激活频率对比
├── discrimination_scores.png     # 判别力得分分布
└── neuron_analysis.txt           # 详细数值结果
```

## 核心评估指标

### 1. 特征激活统计
- **Bonafide sparsity**: 真实语音平均激活多少features
- **Spoof sparsity**: 假语音平均激活多少features
- **期望值**: 3.12% (128/4096) 附近，说明TopK有效

### 2. 判别性评分
```
Top 50 discriminative features identified
Mean discrimination score: 0.xxxx
```
- Score > 0.1: 判别力强
- Score < 0.05: 判别力弱

### 3. 类别特异性neurons
```
Bonafide-specific neurons: XX
Spoof-specific neurons: YY
```
- 越多说明模型学到了更多distinctive patterns
- >10个就算不错

### 4. 总体评分 (0-4分)
- **4分**: 高度可解释，features非常meaningful
- **3分**: 可解释性好，可用于分析
- **2分**: 部分可解释
- **0-1分**: 可解释性有限

## 实用案例

### 案例1: 找出最重要的10个features
从`neuron_analysis.txt`中查看:
```
Top 20 Most Discriminative Features:
  1. Feature 1245: score=0.4521
  2. Feature 3012: score=0.3998
  ...
```

这些features对应到4096维字典空间的特定patterns，可以进一步分析:
- 它们在哪些时间点激活？
- 对应什么acoustic特征？
- 对不同攻击类型的响应如何？

### 案例2: 分析bonafide vs spoof的区别
查看`feature_discrimination.png`:
- 绿色条高: bonafide特有特征（如自然的韵律、呼吸声）
- 红色条高: spoof特有特征（如TTS的机械感、VC的artifacts）

### 案例3: 验证模型没有overfitting到少数features
查看`activation_frequency.png`:
- 如果散点分布广泛: 模型使用diverse features ✓
- 如果集中在少数点: 可能过拟合到某些patterns ✗

## 后续实验方向

### 实验1: 不同sparsity的影响
```bash
# 当前: k=128 (3.12%)
# 对比: k=64 (1.56%), k=256 (6.25%)
```
分析哪个sparsity level提供最好的interpretability

### 实验2: 特征在时序上的变化
修改analyze_sae_neurons.py，保存temporal activation:
```python
# 保存每个时间步的activation
temporal_patterns = interp['sparse_features']  # [B, T, 4096]
```

### 实验3: 不同攻击类型的特征
按攻击算法分组分析（A01-A19）:
```python
# 对每个攻击类型单独统计
tts_features = analyze_attack_type('A01', 'A02', 'A03')
vc_features = analyze_attack_type('A04', 'A05', 'A06')
```

## 论文写作建议

### 可用的论述点
1. **定量证据**: 
   - "Our model achieves X% sparsity with Y discriminative features"
   - "Top-50 features show mean discrimination score of Z"

2. **可视化**:
   - Feature discrimination plots
   - Class-specific neuron heatmaps
   - Temporal activation patterns

3. **对比实验**:
   - 与baseline (无SAE) 对比interpretability
   - 不同SAE配置的feature quality对比

4. **Case studies**:
   - 分析specific examples展示model reasoning
   - 错误案例的feature activation分析

## 常见问题

**Q: 为什么我的模型interpretability score很低？**
A: 可能原因:
- sae_weight太小，reconstruction loss不足
- k值不合适，过大或过小
- 训练不充分，SAE还没学好

**Q: 如何提高feature quality？**
A: 尝试:
- 增大sae_weight (当前0.1，试试0.2, 0.5)
- 调整k值寻找最优sparsity
- 更多training epochs让SAE充分学习

**Q: 分析需要多长时间？**
A: 
- 500 samples: ~5-10分钟 (GPU)
- 1000 samples: ~15-20分钟
- 推荐从500开始，结果足够representative

## 下一步

1. ✓ 等待当前训练完成 (epoch 13/40)
2. ▶ 运行可解释性分析: `sbatch run_sae_analysis.sh`
3. 查看结果，撰写论文interpretability section
4. 实验不同配置对比feature quality

详细说明请参考: `SAE_INTERPRETABILITY_GUIDE.md`
