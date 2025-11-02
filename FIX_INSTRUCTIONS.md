# 🔧 Fix for "TypeError: forward() got an unexpected keyword argument 'return_attention'"

## Problem

Your server's `model_g3_heatmap.py` doesn't have the `return_attention` parameter in the Model class's forward method.

## Quick Solution (Easiest)

### Option 1: Use the updated visualization code (Already Done!)

The `visualize_attention_evaluation.py` file has been updated with a try-except block that will work with both old and new model versions. However, your model still needs to **actually collect** the attention weights.

### Option 2: Modify model_g3_heatmap.py on your server

**Step 1:** SSH to your server and edit the file:

```bash
cd /root/autodl-tmp/SLSforASVspoof-2021-DF
nano model_g3_heatmap.py  # or use vim/vi
```

**Step 2:** Find the Model class's `__init__` method and verify it has this code (should be around line 250):

```python
# Store attention weights for visualization and analysis
self.attention_weights = {
    'temporal': None,
    'intra': None,
    'inter': None
}
```

If it's missing, add it **before** the `self.projection_head` definition.

**Step 3:** Find the `forward` method signature (around line 289):

**CHANGE FROM:**

```python
def forward(self, x, labels=None):
```

**CHANGE TO:**

```python
def forward(self, x, labels=None, return_attention=False):
```

**Step 4:** Inside the forward method, find where temporal attention is computed and ADD this line:

**FIND THIS:**

```python
layer_emb, temporal_attn = self.temporal_attn(layer_tokens)
layer_emb = layer_emb.view(B, L, C)
```

**ADD AFTER:**

```python
# Store temporal attention weights if requested
collect_attn = return_attention or getattr(self, '_collect_attention', False)
if collect_attn:
    self.attention_weights['temporal'] = temporal_attn.view(B, L, T).detach()
```

**Step 5:** Find where intra-group attention is in the loop:

**FIND THIS:**

```python
for g in groups:
    g_vec, intra_attn = self.intra_attn(g)
    g_vec = self.group_refine(g_vec)
    group_vecs.append(g_vec)
```

**CHANGE TO:**

```python
for g in groups:
    g_vec, intra_attn = self.intra_attn(g)
    g_vec = self.group_refine(g_vec)
    group_vecs.append(g_vec)
    if collect_attn:
        intra_attns.append(intra_attn.detach())
```

**AND ADD BEFORE THE LOOP:**

```python
intra_attns = []
```

**AND ADD AFTER THE LOOP:**

```python
# Store intra-group attention weights if requested
if collect_attn and len(intra_attns) > 0:
    self.attention_weights['intra'] = torch.stack(intra_attns, dim=1)
```

**Step 6:** Find where inter-group attention is computed:

**FIND THIS:**

```python
group_stack = torch.stack(group_vecs, dim=1)
utt_emb, inter_attn = self.inter_attn(group_stack)
utt_emb = self.utt_refine(utt_emb)
```

**ADD AFTER:**

```python
# Store inter-group attention weights if requested
if collect_attn:
    self.attention_weights['inter'] = inter_attn.detach()
```

**Step 7:** Save and test:

```bash
# Save the file (Ctrl+O, Enter, Ctrl+X in nano)

# Quick test
python3 -c "
from model_g3_heatmap import Model
import torch

class Args:
    group_size = 3
    use_contrastive = False
    contrastive_weight = 0.1
    supcon_weight = 0.1

model = Model(Args(), 'cpu')
x = torch.randn(2, 64600)
output, _, _, _ = model.forward(x, labels=torch.tensor([0, 1]), return_attention=True)
assert model.attention_weights['temporal'] is not None
print('✅ SUCCESS! Model now supports attention collection!')
"
```

## Alternative: Copy Complete File

I've created a complete replacement forward method in `REPLACEMENT_forward_method.py`. You can:

1. Open `model_g3_heatmap.py` on your server
2. Find the entire `def forward(self, x, labels=None):` method
3. Replace it with the content from `REPLACEMENT_forward_method.py`

## After Fixing

Run your evaluation command again:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /root/autodl-tmp/SLSforASVspoof-2021-DF/models/g3_heatmap_LA_CCE_100_16_1e-06_group3_contrastiveFalse_g3_viz_only/best_model_eer_g3_viz_only.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/release_in_the_wild/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/release_in_the_wild/filenames.txt \
  --track In-the-Wild \
  --viz_dir attention_viz_InTheWild_20251023 \
  --num_viz_samples 100 --batch_size 16 --group_size 3
```

## What Changed in visualize_attention_evaluation.py

The visualization script now:

- ✅ Tries to call forward with `return_attention=True`
- ✅ Falls back gracefully if that parameter doesn't exist
- ✅ Sets `model._collect_attention = True` flag as fallback
- ✅ Provides better error messages
- ✅ Handles evaluation datasets that return utterance IDs instead of labels

You should now be able to generate your attention visualizations! 🎨
