#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting Lightning AI environment setup..."

# 1. Create necessary directories
mkdir -p data
mkdir -p input_video

# 2. Update pip
pip install --upgrade pip

# 3. Install core dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found! Installing default dependencies..."
    pip install torch torchvision transformers ultralytics supervision opencv-python pillow numpy tqdm accelerate bitsandbytes
fi

# 4. Install additional dependencies for Qwen2.5-VL and Molmo
echo "📦 Installing additional specialized dependencies..."
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
pip install yt-dlp  # For YouTube video downloads

# 5. Verify critical installations
echo "🔍 Verifying installations..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "❌ PyTorch failed"
python -c "import bitsandbytes; print(f'✅ BitsAndBytes: {bitsandbytes.__version__}')" || echo "❌ BitsAndBytes failed"
python -c "import yt_dlp; print(f'✅ yt-dlp installed')" || echo "❌ yt-dlp failed"

# 5. Verify bitsandbytes installation (critical for 4-bit)
python -c "import bitsandbytes; print(f'✅ bitsandbytes version: {bitsandbytes.__version__}')"

echo "✅ Setup complete! Please ensure your YOLO models are in the 'data/' directory."
echo "Suggested models:"
echo " - data/football-player-detection.pt"
echo " - data/football-ball-detection.pt"
echo " - data/football-pitch-detection.pt"
