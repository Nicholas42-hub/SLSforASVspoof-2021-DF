# Quick Resume Guide

## TL;DR - Common Usage

### Start Fresh Training

```bash
python main.py --model_type g3_heatmap --track DF --comment my_exp
```

### Resume from Latest Checkpoint (Auto-Find)

```bash
# Just add --resume flag, it will automatically find the latest checkpoint!
python main.py --model_type g3_heatmap --track DF --comment my_exp --resume
```

### Resume from Specific Checkpoint

```bash
python main.py --model_type g3_heatmap --resume --model_path models/.../checkpoint_epoch_50.pth
```

### Force Fresh Start (Even if Checkpoints Exist)

```bash
python main.py --model_type g3_heatmap --track DF --comment my_exp --fresh_start
```

---

## How `--resume` Works Now

### Option 1: Auto-Find Latest Checkpoint (Recommended)

```bash
python -u main.py \
--database_path /root/autodl-tmp/CLAD/Datasets/LA/ \
--protocols_path /root/autodl-tmp/CLAD/Datasets/LA/ \
--model_type g3_heatmap \
--batch_size 16 \
--num_epochs 100 \
--lr 1e-6 \
--weight_decay 1e-4 \
--group_size 3 \
--visualize_attention \
--viz_samples 30 \
--viz_frequency 1 \
--track DF \
--comment g3_viz_only \
--resume
```

**What happens:**

1. ✅ Automatically builds the model save path based on your arguments
2. ✅ Searches for the latest checkpoint in that directory
3. ✅ If found: Resumes from that checkpoint
4. ✅ If not found: Starts fresh training (no error)

### Option 2: Specify Exact Checkpoint

```bash
python -u main.py \
--model_type g3_heatmap \
--resume \
--model_path models/g3_heatmap_DF_CCE_100_16_1e-06_group3_contrastiveFalse_g3_viz_only/checkpoint_epoch_42_g3_viz_only.pth \
--num_epochs 100
```

**What happens:**

1. ✅ Loads the exact checkpoint you specified
2. ✅ Resumes training from that specific epoch

---

## Comparison Table

| Command                                       | Behavior                                                   |
| --------------------------------------------- | ---------------------------------------------------------- |
| `python main.py --comment exp1`               | Fresh start, epoch 1 → 100                                 |
| `python main.py --comment exp1 --resume`      | Auto-find latest checkpoint in `exp1` dir, resume if found |
| `python main.py --comment exp1 --fresh_start` | Fresh start, ignore any checkpoints                        |
| `python main.py --resume --model_path <path>` | Resume from specific checkpoint                            |
| `python main.py --auto_resume --comment exp1` | Same as `--resume` (auto-find and resume)                  |

---

## Your Exact Command

```bash
python -u main_single_loss.py \
--database_path /root/autodl-tmp/CLAD/Datasets/LA/ \
--protocols_path /root/autodl-tmp/CLAD/Datasets/LA/ \
--model_type g3_heatmap \
--batch_size 16 \
--num_epochs 100 \
--lr 1e-6 \
--weight_decay 1e-4 \
--group_size 3 \
--visualize_attention \
--viz_samples 30 \
--viz_frequency 1 \
--track DF \
--comment g3_viz_only \
--resume
```

**This will:**

1. Look in: `models/g3_heatmap_DF_CCE_100_16_1e-06_group3_contrastiveFalse_g3_viz_only/`
2. Find the latest `checkpoint_epoch_*.pth` file
3. Resume from that checkpoint (restore epoch, optimizer, best EER)
4. Continue training to epoch 100

**First time running (no checkpoint):**

- Starts fresh from epoch 1
- Saves checkpoints as you train

**Second time running (checkpoint exists):**

- Automatically resumes from latest checkpoint
- Continues from where it left off

---

## Example Workflow

```bash
# Day 1: Start training
python main.py --model_type g3_heatmap --track DF --comment day1

# Process killed at epoch 25...

# Day 2: Just add --resume!
python main.py --model_type g3_heatmap --track DF --comment day1 --resume
# ✅ Auto-resumes from epoch 25

# Training completes at epoch 100

# Day 3: Want to train more epochs
python main.py --model_type g3_heatmap --track DF --comment day1 --num_epochs 150 --resume
# ✅ Auto-resumes from epoch 100, trains to 150
```

---

## Messages You'll See

### When Checkpoint Found

```
🔍 --resume flag: Auto-finding latest checkpoint at epoch 42
✅ RESUMING TRAINING from checkpoint
   📦 Loaded model weights + optimizer state
   📍 Checkpoint from epoch: 42
   🔄 Will resume from epoch: 43
   📊 Best validation EER so far: 0.16%
```

### When No Checkpoint Found

```
⚠️  --resume flag specified but no checkpoint found in models/.../
   Starting fresh training instead...
✨ FRESH TRAINING START
📍 Starting from epoch: 1
🎯 Total epochs: 100
```

---

## Pro Tips

1. **Always use `--comment`** to organize experiments

   ```bash
   --comment exp1_baseline
   --comment exp2_more_data
   ```

2. **Resume is safe to use always**

   ```bash
   # This works whether checkpoint exists or not
   python main.py --model_type g3_heatmap --resume --comment my_exp
   ```

3. **Check what will be resumed**

   ```bash
   python main.py --list_checkpoints --comment my_exp
   ```

4. **Start fresh even if checkpoints exist**
   ```bash
   python main.py --fresh_start --comment my_exp
   ```

---

## Summary

✅ `--resume` now **automatically finds** the latest checkpoint  
✅ No need to specify `--model_path` manually  
✅ Safe to use - won't error if no checkpoint exists  
✅ Still supports `--model_path` for specific checkpoints  
✅ Use `--fresh_start` to explicitly ignore checkpoints

**Your command is now simpler:**

```bash
# Before (manual path)
--resume --model_path models/long/path/to/checkpoint_epoch_42.pth

# Now (auto-find)
--resume
```
