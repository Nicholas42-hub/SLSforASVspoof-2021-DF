# Patch Instructions for model_g3_heatmap.py

## Problem

The `model_g3_heatmap.py` file on your server doesn't support the `return_attention` parameter in the forward method, which is needed for attention visualization.

## Solution

You need to modify the `forward` method in the `Model` class in `model_g3_heatmap.py`.

### Find this line (around line 289):

```python
def forward(self, x, labels=None):
```

### Replace with:

```python
def forward(self, x, labels=None, return_attention=False):
```

### Then, in the forward method, find where temporal attention is computed and add:

```python
# After: layer_emb, temporal_attn = self.temporal_attn(layer_tokens)
if return_attention or getattr(self, '_collect_attention', False):
    self.attention_weights['temporal'] = temporal_attn.view(B, L, T).detach()
```

### Find where intra-group attention is collected and add:

```python
# After: g_vec, intra_attn = self.intra_attn(g)
if return_attention or getattr(self, '_collect_attention', False):
    intra_attns.append(intra_attn.detach())

# After the loop, before inter_attn:
if (return_attention or getattr(self, '_collect_attention', False)) and len(intra_attns) > 0:
    self.attention_weights['intra'] = torch.stack(intra_attns, dim=1)
```

### Find where inter-group attention is computed and add:

```python
# After: utt_emb, inter_attn = self.inter_attn(group_stack)
if return_attention or getattr(self, '_collect_attention', False):
    self.attention_weights['inter'] = inter_attn.detach()
```

## Alternative: Quick Fix

If the above is too complex, you can simply **always collect attention weights** by removing all the `if return_attention` checks and always storing the attention weights. This has minimal performance impact.

## Verification

After making changes, run this to verify:

```python
import torch
from model_g3_heatmap import Model

# Create dummy args
class Args:
    group_size = 3
    use_contrastive = False

model = Model(Args(), 'cpu')
x = torch.randn(2, 64600)
labels = torch.tensor([0, 1])

# Test with return_attention
output, _, _, _ = model.forward(x, labels=labels, return_attention=True)

# Check if attention weights are collected
assert model.attention_weights['temporal'] is not None, "Temporal attention not collected!"
print("✅ Model supports attention collection!")
```
