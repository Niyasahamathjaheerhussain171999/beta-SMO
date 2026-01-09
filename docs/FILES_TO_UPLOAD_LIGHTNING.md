# 📁 Files to Upload to Lightning AI

## Required Files (MUST UPLOAD)

### 1. Main Code Files
```
smo-model-niyas/
├── main4.py                    ✅ REQUIRED
├── shot_detection.py           ✅ REQUIRED
├── requirements.txt            ✅ REQUIRED
├── setup_lightning.sh          ✅ REQUIRED
└── scout_report_generator.py   ✅ REQUIRED (if you have it)
```

### 2. YOLO Model Files (in `data/` folder)
```
data/
├── football-player-detection.pt   ✅ REQUIRED
├── football-ball-detection.pt     ✅ REQUIRED
└── football-pitch-detection.pt     ✅ REQUIRED (optional but recommended)
```

### 3. Video Files (in `input_video/` folder)
```
input_video/
└── video_segment (1).mp4      ✅ Your video file
```

---

## Optional Files (Nice to Have)

### Documentation (Helpful but not required)
```
├── SHOT_DETECTION_EXPLAINED.md
├── SHOT_DETECTION_FIXES.md
├── LIGHTNING_AI_QUICK_START.md
└── COPY_PASTE_COMMANDS.txt
```

### Other Scripts (Optional)
```
├── fix_video_format.py
└── README.md
```

---

## Quick Upload Checklist

### ✅ Minimum Required:
1. `main4.py`
2. `shot_detection.py`
3. `requirements.txt`
4. `setup_lightning.sh`
5. `data/football-player-detection.pt`
6. `data/football-ball-detection.pt`
7. `data/football-pitch-detection.pt` (optional)
8. `input_video/video_segment (1).mp4` (your video)

### 📦 Recommended (Full Project):
Upload the entire `smo-model-niyas/` folder - this includes everything!

---

## How to Upload

### Method 1: Upload Entire Folder (Easiest)
1. In Lightning AI, click "Upload" button
2. Select entire `smo-model-niyas/` folder
3. Wait for upload to complete

### Method 2: Upload Individual Files
1. Create folder structure:
   ```
   smo-model-niyas/
   ├── data/
   └── input_video/
   ```
2. Upload files to correct locations

---

## File Structure After Upload

Your Lightning AI workspace should look like:
```
workspace/
└── smo-model-niyas/
    ├── main4.py
    ├── shot_detection.py
    ├── requirements.txt
    ├── setup_lightning.sh
    ├── data/
    │   ├── football-player-detection.pt
    │   ├── football-ball-detection.pt
    │   └── football-pitch-detection.pt
    └── input_video/
        └── video_segment (1).mp4
```

---

## Verify Upload

After uploading, check files exist:
```bash
ls -la smo-model-niyas/
ls -la smo-model-niyas/data/
ls -la smo-model-niyas/input_video/
```

---

## That's It!

Once files are uploaded, just run:
```bash
bash setup_lightning.sh
python main4.py --source "input_video/video_segment (1).mp4" --debug
```


