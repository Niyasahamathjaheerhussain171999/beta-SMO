# 🚀 SIMPLE STEPS - What To Do

## Step 1: Open Lightning AI
1. Go to https://lightning.ai
2. Sign in (or create account)
3. Click "Create Studio"
4. Choose a GPU (pick A10G or A100 - the big one)

## Step 2: Upload Your Files
1. In Lightning Studio, click "Upload" button
2. Upload your `smo-model-niyas` folder
3. Wait for upload to finish

## Step 3: Open Terminal
1. In Lightning Studio, click "Terminal" tab (bottom of screen)
2. You'll see a command prompt

## Step 4: Run Setup (Copy & Paste This)
```bash
bash setup_lightning.sh
```
Press Enter. Wait for it to finish (takes 2-5 minutes)

## Step 5: Download YouTube Video & Analyze
Copy and paste this command:
```bash
python main4.py --source "https://youtu.be/osoyzB5UR88?si=l9m09maTsbWjOF6y" --debug
```
Press Enter. This will:
- Download the video in 1080p HD
- Analyze it for shots (on target/off target)
- Create reports

## Step 6: Wait
- First time: Takes 5-10 minutes (downloading models)
- After that: Takes 1-5 minutes per minute of video
- You'll see progress messages

## Step 7: Get Results
After it finishes, you'll see files created:
- `shot_report.json` - Shot data
- `shot_report.csv` - Shot data (Excel format)
- `scout_report.html` - Click to open in browser
- `annotated_test_video.mp4` - Video with shot markers

## That's It! 🎉

**Need help?** Check the console output - it shows what's happening.


