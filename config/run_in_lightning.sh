#!/bin/bash
# Simple script to run shot detection in Lightning AI
# Just copy and paste this entire script into Lightning AI terminal

echo "🚀 Starting Shot Detection Analysis..."

# Navigate to project directory (adjust if needed)
cd smo-model-niyas 2>/dev/null || echo "Already in smo-model-niyas directory"

# Run the analysis
# Change the video path below to your video file
python main4.py --source "input_video/video_segment (1).mp4" --debug

echo "✅ Analysis complete! Check output files:"
echo "   - shot_report.json"
echo "   - shot_report.csv"
echo "   - scout_report.html"
echo "   - annotated_test_video.mp4"


