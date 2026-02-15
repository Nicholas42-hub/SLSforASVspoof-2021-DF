"""Quick test with 2000 samples to verify logic."""
import torch
import torch.nn as nn
import numpy as np
import os
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from model_window_topk import Model as WindowTopKModel
import json
from collections import defaultdict


def pad(x, max_len=64600):
    """Pad or truncate audio to fixed length"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2021_eval_torchaudio(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        flac_path = os.path.join(self.base_dir, 'flac', f'{utt_id}.flac')
        
        try:
            waveform, sample_rate = torchaudio.load(flac_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            X = waveform.squeeze().numpy()
            X_pad = pad(X, self.cut)
            x_inp = Tensor(X_pad)
            return x_inp, utt_id
        except Exception as e:
            print(f"Warning: Failed to load {utt_id}: {e}")
            x_inp = Tensor(np.zeros(self.cut))
            return x_inp, utt_id

print('='*60)
print('QUICK TEST: 2000 samples to verify logic')
print('='*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load model
class DummyArgs:
    pass

model = WindowTopKModel(
    args=DummyArgs(),
    device=device,
    use_sae=True,
    sae_dict_size=4096,
    sae_k=128,
    sae_window_size=8,
    sae_weight=0.1
)

checkpoint = torch.load('models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth', map_location=device)
model = nn.DataParallel(model).to(device)
model.load_state_dict(checkpoint, strict=False)
model.eval()
print('✓ Model loaded')

# Load top-k features
with open('decision_analysis_2021LA_5k/decision_analysis_results.json', 'r') as f:
    decision_results = json.load(f)
top_k_features = set(decision_results['attribution']['top_k_indices'][:50])
print(f'✓ Top-50 features loaded')

# Load dataset
database_path = '/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval/'

# Get first 2000 FLAC files from directory  
flac_dir = os.path.join(database_path, 'flac')
all_files = sorted([f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')])
file_list = all_files[:2000]

print(f'✓ Found {len(all_files)} total files, using first 2000')

eval_dataset = Dataset_ASVspoof2021_eval_torchaudio(list_IDs=file_list, base_dir=database_path)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)

# Analyze 2000 samples
all_feature_lifetimes = defaultdict(list)
count = 0

with torch.no_grad():
    for batch_x, _ in eval_loader:
        if count >= 2000:
            break
        batch_x = batch_x.to(device)
        output = model(batch_x, return_sae_loss=False, return_interpretability=True)
        sparse_features = model.module.last_sparse_features
        
        for b in range(sparse_features.shape[0]):
            if count >= 2000:
                break
            features = sparse_features[b]
            active_mask = (features > 0)
            T, D = active_mask.shape
            
            # Compute lifetimes
            for d in range(D):
                feature_active = active_mask[:, d]
                current_run = 0
                for t in range(T):
                    if feature_active[t]:
                        current_run += 1
                    else:
                        if current_run > 0:
                            all_feature_lifetimes[d].append(current_run)
                            current_run = 0
                if current_run > 0:
                    all_feature_lifetimes[d].append(current_run)
            
            count += 1
            if count % 100 == 0: print(f"  Processed {count}/2000 samples")

print(f'\n✓ Analyzed {count} samples')
print(f'✓ Features with activity: {len(all_feature_lifetimes)}')

# Compute stats
discriminative_lifetimes = []
non_discriminative_lifetimes = []

for fid, lifetimes in all_feature_lifetimes.items():
    if fid in top_k_features:
        discriminative_lifetimes.extend(lifetimes)
    else:
        non_discriminative_lifetimes.extend(lifetimes)

print(f'\n' + '='*60)
print('RESULTS')
print('='*60)
print(f'Discriminative features (top-50):')
print(f'  Lifetime instances: {len(discriminative_lifetimes)}')
if discriminative_lifetimes:
    print(f'  Mean lifetime: {np.mean(discriminative_lifetimes):.2f} timesteps')
    print(f'  Median lifetime: {np.median(discriminative_lifetimes):.2f} timesteps')

print(f'\nNon-discriminative features:')
print(f'  Lifetime instances: {len(non_discriminative_lifetimes)}')
if non_discriminative_lifetimes:
    print(f'  Mean lifetime: {np.mean(non_discriminative_lifetimes):.2f} timesteps')
    print(f'  Median lifetime: {np.median(non_discriminative_lifetimes):.2f} timesteps')

if discriminative_lifetimes and non_discriminative_lifetimes:
    diff = np.mean(discriminative_lifetimes) - np.mean(non_discriminative_lifetimes)
    print(f'\nDifference: {diff:.2f} timesteps')
    if diff < 0:
        print('✅ Discriminative features ARE more transient (shorter lifetime)')
        print('   This supports the hypothesis!')
    else:
        print('❌ Discriminative features are MORE persistent (longer lifetime)')
        print('   Hypothesis not supported')
    print('\n✅ Logic verified! Ready for full 2000-sample run.')
else:
    print('\n⚠️ Insufficient data in 2000 samples')

print('='*60)
