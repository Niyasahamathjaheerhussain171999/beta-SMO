# ⚡ Lightning AI - Quick Start Guide

## Step 1: Setup (Run Once - Copy & Paste All)

Open Terminal in Lightning AI and paste this:

```bash
# Create directories
mkdir -p data input_video

# Install dependencies
pip install --upgrade pip
pip install torch torchvision transformers ultralytics supervision opencv-python pillow numpy tqdm accelerate bitsandbytes
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
pip install yt-dlp

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
```

**Wait for installation to finish** (takes 2-5 minutes)

---

## Step 2: Upload Your Files

### Option A: Upload via Lightning AI UI
1. Click "Upload" button in Lightning AI
2. Upload these files/folders:
   - `smo-model-niyas/` folder (entire folder)
   - YOLO models to `data/` folder:
     - `data/football-player-detection.pt`
     - `data/football-ball-detection.pt`
     - `data/football-pitch-detection.pt`
   - Video files to `input_video/` folder

### Option B: Use Git (if your code is on GitHub)
```bash
git clone <your-repo-url>
cd <repo-name>
```

---

## Step 3: Run Analysis (Copy & Paste)

### For Local Video File:
```bash
cd smo-model-niyas
python main4.py --source "input_video/your_video.mp4" --debug
```

### For YouTube Video:
```bash
cd smo-model-niyas
python main4.py --source "https://youtu.be/osoyzB5UR88?si=l9m09maTsbWjOF6y" --debug
```

### For Your Specific Video:
```bash
cd smo-model-niyas
python main4.py --source "input_video/video_segment (1).mp4" --debug
```

---

## What Happens Next

1. **Models Load** (first time: 5-10 minutes)
   - Player detection model
   - Ball detection model
   - Pitch detection model
   - Molmo-7B VLM
   - Qwen2.5-VL VLM

2. **Video Processes** (1-5 minutes per minute of video)
   - Detects players and ball
   - Tracks movements
   - Analyzes passes and shots
   - Shows progress bar

3. **Results Generated**
   - `shot_report.json` - Shot data
   - `shot_report.csv` - Shot data (Excel format)
   - `scout_report.html` - Interactive report
   - `annotated_test_video.mp4` - Video with annotations

---

## Complete Example (Copy All at Once)

```bash
# Navigate to project
cd smo-model-niyas

# Run analysis on your video
python main4.py --source "input_video/video_segment (1).mp4" --debug
```

That's it! Just copy and paste the command above.

---

## Troubleshooting

### "Module not found" error?
```bash
pip install <module-name>
```

### "Model file not found" error?
- Make sure YOLO model files are in `data/` folder
- Check file names match exactly

### "Video file not found" error?
- Make sure video is in `input_video/` folder
- Check file name matches exactly (case-sensitive)

### Out of Memory (OOM) error?
- Use a GPU with more VRAM (A100 instead of A10G)
- Or reduce video resolution

---

## Quick Commands Reference

```bash
# Setup (first time only)
bash setup_lightning.sh

# Run analysis
python main4.py --source "input_video/video.mp4" --debug

# Run without VLM (faster, less accurate)
python main4.py --source "input_video/video.mp4" --no-vlm

# Download YouTube video first, then analyze
python main4.py --source "https://youtube.com/watch?v=..." --debug
```

---

## Files You'll Get

After running, check these files:

- `shot_report.json` - All shot data
- `shot_report.csv` - Shot data for Excel
- `scout_report.html` - Open in browser to view
- `annotated_test_video.mp4` - Video with shot markers
- `match_analysis_result.json` - Pass analysis data

---

## That's It!

Just copy and paste:
```bash
cd smo-model-niyas
python main4.py --source "input_video/video_segment (1).mp4" --debug
```

Wait for it to finish, then check the output files! 🎉


