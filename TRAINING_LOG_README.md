# 训练日志和实验管理说明

## 新功能

1. **CSV 日志记录**: 每个 epoch 的训练指标会自动保存到 CSV 文件
2. **Comment 参数**: 用于区分不同的实验，会包含在文件夹名和模型文件名中

## 使用方法

### 基本训练命令

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --track=DF --lr=0.000001 --batch_size=16 --loss=weighted_CCE --num_epochs=100 --comment="baseline_experiment"
```

### 命令行参数

- `--comment`: 实验注释，用于区分不同的实验。会添加到模型文件夹名和模型文件名中

## 输出文件结构

当你运行训练时，会创建如下的文件结构：

```
models/
├── model_DF_weighted_CCE_100_16_1e-06_baseline_experiment/
│   ├── training_log.csv                 # 每个epoch的训练日志
│   ├── best_model_eer_baseline_experiment.pth  # 最佳模型（基于EER）
│   ├── epoch_0_baseline_experiment.pth  # 第0个epoch的模型
│   ├── epoch_1_baseline_experiment.pth  # 第1个epoch的模型
│   └── ...
```

## CSV 日志格式

`training_log.csv` 包含以下列：

- `epoch`: 训练轮数
- `timestamp`: 时间戳
- `train_loss`: 训练损失
- `val_loss`: 验证损失
- `val_acc`: 验证准确率
- `val_eer`: 验证等错误率 (Equal Error Rate)

## 示例使用场景

### 不同学习率实验

```bash
# 实验1: 学习率 1e-6
python main.py --comment="lr_1e6" --lr=0.000001

# 实验2: 学习率 1e-5
python main.py --comment="lr_1e5" --lr=0.00001

# 实验3: 学习率 1e-4
python main.py --comment="lr_1e4" --lr=0.0001
```

### 不同批大小实验

```bash
# 实验1: batch size 8
python main.py --comment="bs8" --batch_size=8

# 实验2: batch size 16
python main.py --comment="bs16" --batch_size=16

# 实验3: batch size 32
python main.py --comment="bs32" --batch_size=32
```

### 消融研究

```bash
# 基线模型
python main.py --comment="baseline"

# 添加某个功能的模型
python main.py --comment="with_feature_A"

# 添加另一个功能的模型
python main.py --comment="with_feature_B"
```

## 日志分析

你可以使用 Python 来分析训练日志：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
df = pd.read_csv('models/model_DF_weighted_CCE_100_16_1e-06_baseline_experiment/training_log.csv')

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['val_acc'])
plt.title('Validation Accuracy')

plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['val_eer'])
plt.title('Validation EER')

plt.tight_layout()
plt.show()
```
