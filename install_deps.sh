#!/bin/bash
# Install missing dependencies for TopK SAE project

source SLS_venv/bin/activate

echo "Installing missing dependencies..."

pip install librosa==0.9.2
pip install tensorboardX
pip install tqdm
pip install soundfile
pip install scipy
pip install scikit-learn

echo "✓ Dependencies installed"
