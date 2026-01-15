#!/bin/bash
# ScoutMe Video Downloader for Lightning AI
# This script downloads the Veo match video directly to your A100 environment.

echo "🎬 Starting video download for ScoutMe Analysis..."
echo "🔗 Source: Veo App"

# Ensure yt-dlp is installed
pip install -q yt-dlp

# Video URL
VIDEO_URL="https://app.veo.co/matches/20201123-20201123164449-656b6fa0/"

# Create input_video directory if it doesn't exist
mkdir -p "input_video"

# Download command
# We use -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' to ensure mp4 format
yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' \
       -o "input_video/veo_match_video.mp4" \
       "$VIDEO_URL"

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS: veo_match_video.mp4 downloaded to input_video/ !"
    echo "📂 File size: $(du -sh input_video/veo_match_video.mp4 | cut -f1)"
else
    echo "❌ ERROR: Download failed. Please check your internet connection or the URL."
fi
