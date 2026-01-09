"""
Fix video format - Convert HLS/MPEG-TS MP4 to proper MP4 format
"""
import cv2
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

input_video = "input_video/youtube_osoyzB5UR88.mp4"
output_video = "input_video/youtube_osoyzB5UR88_fixed.mp4"

print("Fixing video format...")
print(f"   Input: {input_video}")
print(f"   Output: {output_video}")

# Open input video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ Error: Cannot open input video")
    exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"   Resolution: {width}x{height}")
print(f"   FPS: {fps}")
print(f"   Frames: {total_frames}")

# Create video writer with proper codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    out.write(frame)
    frame_count += 1
    
    if frame_count % 100 == 0:
        print(f"   Processing: {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")

cap.release()
out.release()

print(f"Video fixed! Saved to: {output_video}")
print(f"   Processed {frame_count} frames")

