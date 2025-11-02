# Checkpoint Management Guide

This guide explains how to use the checkpoint management arguments in `main.py`.

## New Arguments

### `--resume`

Resume training from a checkpoint, restoring optimizer state and epoch counter.

**Requirements:** Must be used with `--model_path`

**Example:**

```bash
python main.py \
--model_type g3_heatmap \
--resume \
--model_path models/g3_heatmap_DF_CCE_100_16_1e-06_group3_contrastiveFalse/checkpoint_epoch_10.pth \
--num_epochs 100
```

### `--fresh_start`

Start fresh training from epoch 0, ignoring any existing checkpoints.

**Use cases:**

- Starting a completely new experiment
- Resetting training even if checkpoints exist
- Loading pre-trained weights but starting epoch counter from 0

**Example:**

```bash
python main.py \
--model_type g3_heatmap \
--fresh_start \
--num_epochs 100 \
--comment new_experiment
```

**With pre-trained weights:**

```bash
python main.py \
--model_type g3_heatmap \
--fresh_start \
--model_path models/pretrained/best_model_eer.pth \
--num_epochs 100
```

### `--auto_resume`

Automatically find and resume from the latest checkpoint if available.

**How it works:**

- Searches for the latest checkpoint in the model save directory
- If found, automatically sets `--resume` and loads the checkpoint
- If not found, starts fresh training

**Example:**

```bash
python main.py \
--model_type g3_heatmap \
--auto_resume \
--num_epochs 100 \
--comment my_experiment
```

### `--list_checkpoints`

List all available checkpoints with their metrics and exit.

**Example:**

```bash
python main.py --list_checkpoints --comment my_experiment
```

**Output:**

```
======================================================================
📋 Available Checkpoints in models/g3_heatmap_DF_CCE_100_16_1e-06_group3_contrastiveFalse_my_experiment
======================================================================

📦 checkpoint_epoch_10_my_experiment.pth
   Epoch: 10
   Val EER: 0.16%
   Val Acc: 99.84%
   Best EER: 0.16%

📦 checkpoint_epoch_20_my_experiment.pth
   Epoch: 20
   Val EER: 0.12%
   Val Acc: 99.88%
   Best EER: 0.12%

📦 best_checkpoint_eer_my_experiment.pth
   Epoch: 20
   Val EER: 0.12%
   Val Acc: 99.88%
   Best EER: 0.12%
======================================================================
```

## Usage Scenarios

### Scenario 1: Fresh Training (No Checkpoints)

```bash
python main.py \
--model_type g3_heatmap \
--batch_size 16 \
--num_epochs 100 \
--comment exp1
```

**Result:** Starts from epoch 1, trains to epoch 100

### Scenario 2: Resume After Interruption

Training stopped at epoch 50, want to continue to epoch 100:

```bash
python main.py \
--model_type g3_heatmap \
--resume \
--model_path models/g3_heatmap_DF_CCE_100_16_1e-06_group3_contrastiveFalse_exp1/checkpoint_epoch_50_exp1.pth \
--num_epochs 100 \
--comment exp1
```

**Result:** Resumes from epoch 51, trains to epoch 100

### Scenario 3: Continue Training Beyond Original Target

Finished 100 epochs, want to train 50 more:

```bash
python main.py \
--model_type g3_heatmap \
--resume \
--model_path models/g3_heatmap_DF_CCE_100_16_1e-06_group3_contrastiveFalse_exp1/checkpoint_epoch_100_exp1.pth \
--num_epochs 150 \
--comment exp1_extended
```

**Result:** Resumes from epoch 101, trains to epoch 150

### Scenario 4: Use Pre-trained Weights (Fresh Start)

Want to use weights from a trained model but start epoch counter from 0:

```bash
python main.py \
--model_type g3_heatmap \
--fresh_start \
--model_path models/exp1/best_model_eer.pth \
--num_epochs 100 \
--comment exp2_from_pretrained
```

**Result:** Loads model weights, starts from epoch 1 with fresh optimizer

### Scenario 5: Auto-Resume (Convenient for Scripts)

Use this when you want to automatically continue if interrupted:

```bash
python main.py \
--model_type g3_heatmap \
--auto_resume \
--num_epochs 100 \
--comment exp_auto
```

**Result:**

- First run: Starts fresh from epoch 1
- After interruption: Automatically resumes from latest checkpoint
- No need to manually specify checkpoint path

### Scenario 6: Check Progress Before Resuming

```bash
# 1. List available checkpoints
python main.py --list_checkpoints --comment exp1

# 2. Choose a specific checkpoint to resume
python main.py \
--model_type g3_heatmap \
--resume \
--model_path models/.../checkpoint_epoch_75_exp1.pth \
--num_epochs 100 \
--comment exp1
```

## Argument Combinations

### ✅ Valid Combinations

| Combination                         | Behavior                              |
| ----------------------------------- | ------------------------------------- |
| No arguments                        | Fresh start                           |
| `--fresh_start`                     | Fresh start (explicit)                |
| `--resume --model_path <path>`      | Resume from checkpoint                |
| `--auto_resume`                     | Auto-detect and resume or fresh start |
| `--model_path <path>` (no flags)    | Load weights, fresh start             |
| `--fresh_start --model_path <path>` | Load weights, fresh start (explicit)  |

### ❌ Invalid Combinations

| Combination                         | Error                           |
| ----------------------------------- | ------------------------------- |
| `--resume --fresh_start`            | Cannot use both flags           |
| `--resume` (without `--model_path`) | Resume requires checkpoint path |

## Checkpoint Contents

### Full Checkpoint (checkpoint*epoch*\*.pth)

```python
{
    'epoch': 10,                      # Last completed epoch
    'model_state_dict': {...},        # Model weights
    'optimizer_state_dict': {...},    # Optimizer state (momentum, etc.)
    'train_loss': 0.154158,          # Last training loss
    'train_eer': 7.29,               # Last training EER
    'val_loss': 0.005543,            # Last validation loss
    'val_acc': 99.84,                # Last validation accuracy
    'val_eer': 0.16,                 # Last validation EER
    'best_val_eer': 0.16,            # Best EER seen so far
    'args': {...}                     # All hyperparameters
}
```

### Legacy Checkpoint (older format)

```python
{
    # Only contains model_state_dict
    # Cannot resume training state
}
```

## Tips

1. **Regular Checkpoints:** Every epoch saves `checkpoint_epoch_N.pth`
2. **Best Checkpoints:** Best model based on EER saves `best_checkpoint_eer.pth`
3. **Automatic Cleanup:** Consider deleting intermediate checkpoints to save space
4. **Hyperparameter Changes:** Warning shown if hyperparameters differ from checkpoint
5. **Visualization:** Checkpoints work with `--visualize_attention` flag

## Example Workflow

```bash
# Day 1: Start training
python main.py --model_type g3_heatmap --num_epochs 100 --comment day1

# Interruption at epoch 30...

# Day 2: Resume training
python main.py --model_type g3_heatmap --resume \
--model_path models/.../checkpoint_epoch_30_day1.pth \
--num_epochs 100 --comment day1

# After reaching 100 epochs, want to train more
python main.py --model_type g3_heatmap --resume \
--model_path models/.../checkpoint_epoch_100_day1.pth \
--num_epochs 150 --comment day1_extended

# Start a new experiment with best weights from day1
python main.py --model_type g3_heatmap --fresh_start \
--model_path models/.../best_model_eer_day1.pth \
--num_epochs 100 --comment day3_new_exp
```

## Recommended Practices

1. **Use `--comment`** to easily identify experiments
2. **Use `--auto_resume`** for long training jobs that might be interrupted
3. **Use `--list_checkpoints`** to verify available checkpoints before resuming
4. **Use `--fresh_start`** explicitly when starting a new experiment to avoid confusion
5. **Keep best checkpoints**, delete intermediate ones after training completes
