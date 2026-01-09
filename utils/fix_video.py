import cv2
import os

input_path = "input_video/drogba_goal.mp4"
output_path = "input_video/drogba_goal_playable.mp4"

print(f"Fixing video file: {input_path}")

try:
    # Open source video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open source video.")
        exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0  # Default fallback
    
    print(f"   Original: {width}x{height} @ {fps}fps")

    # Define codec and create VideoWriter
    # mp4v is generally widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"   Processed {frame_count} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\nVideo fixed successfully: {output_path}")
    print(f"   Total frames: {frame_count}")
    
except Exception as e:
    print(f"Error fixing video: {e}")
