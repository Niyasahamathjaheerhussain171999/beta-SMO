# Shot Detection Improvements - Summary

## Overview
This document describes the improvements made to the shot detection system to ensure accurate detection and classification of shots on goal (on target vs off target).

## Key Improvements Made

### 1. **Fixed Frame Buffer Passing** ✅
- **Issue**: Frame buffer was not being properly extracted from temporal_buffer
- **Fix**: Correctly extracts frames from temporal_buffer.frames deque
- **Location**: `main4.py` line ~1474

### 2. **Improved Detection Sensitivity** ✅
- **Velocity Threshold**: Lowered from 10 to 8 px/frame (catches slower shots)
- **Confidence Threshold**: Lowered from 60% to 55% (accepts more valid shots)
- **Goal Proximity Radius**: Increased from 100m to 120m (catches shots from further out)
- **Cooldown**: Reduced from 25 to 20 frames (detects shots closer together)
- **Angle Tolerance**: Increased from 135° to 150° (more permissive trajectory check)

### 3. **Enhanced VLM Prompts** ✅
- **Stage 1 (Shot Detection)**: More detailed prompts with context about ball speed and distance
- **Stage 2 (Shot Classification)**: Improved trajectory-based analysis with angle information
- **Video Layer Prompt**: Enhanced with context data (speed, distance, angle) for Qwen2.5-VL

### 4. **Better Debugging Output** ✅
- More frequent debug logging (every 150 frames instead of 300)
- Shows speed, distance, angle, and vector intersection status
- Lower threshold for debug output (speed >= 6 instead of 12)

### 5. **Video Annotations** ✅
- Shots are now drawn on the annotated video with color coding:
  - **Gold**: Goals
  - **Green**: Shots on target
  - **Red**: Shots off target
  - **Orange**: Shots blocked
- Shot count displayed in video overlay

## How Shot Detection Works

### Detection Pipeline

1. **Geometric Filters**:
   - Ball velocity check (must be >= 8 px/frame)
   - Shooting range check (player in valid area)
   - Distance to goal check (must be within 120m)
   - Velocity vector intersection (ball trajectory toward goal, angle < 150°)

2. **AI Analysis** (if geometric filters pass):
   - **Video Layer** (Qwen2.5-VL): Analyzes last 75 frames (~3 seconds) for motion context
   - **Static Layer** (Molmo-7B): Analyzes single frame crop if Video Layer unavailable
   - **Stage 1**: Determines if action is a shot (vs pass/cross/clearance)
   - **Stage 2**: Classifies shot outcome (on target, off target, blocked, goal)

3. **Classification Rules**:
   - **Shot On Target**: Ball trajectory will enter goal frame (between posts, under crossbar)
   - **Shot Off Target**: Ball trajectory will miss goal (over bar or wide of posts)
   - **Shot Blocked**: Defender intercepts the ball
   - **Goal**: Ball crosses goal line into net

## Output Files Generated

### 1. **shot_report.json**
Contains:
- Summary statistics (total shots, on target, off target, blocked, goals)
- Team breakdown
- Individual shot events with:
  - Time, frame number
  - Shooter ID and team
  - Shot type and confidence
  - Ball and shooter positions
  - Enhanced metrics (velocity, angle, distance)

### 2. **shot_report.csv**
CSV format with all shot event data for easy analysis in Excel/Google Sheets

### 3. **Annotated Video**
Video with visual markers showing:
- Shot locations (colored circles)
- Shot type labels
- Shot count overlay

### 4. **Combined HTML Report** (if scout_report_generator.py available)
Interactive HTML report with:
- Shot timeline
- Clickable shot events
- Video playback with markers
- Statistics dashboard

## Usage Tips

### To Get Better Results:

1. **Enable Debug Mode**: Run with `debug=True` to see detailed detection logs
   ```python
   run_analysis(video_path, use_vlm=True, debug=True)
   ```

2. **Check Debug Output**: Look for messages like:
   - `[Shot Check]` - Shot detection triggered
   - `[Shot Filter]` - Shot filtered out (shows why)
   - `[Shot Debug]` - Periodic status of potential shots

3. **Adjust Thresholds** (if needed):
   - Edit `shot_detection.py` parameters:
     - `SHOT_VELOCITY_THRESHOLD`: Lower = more sensitive
     - `SHOT_CONFIDENCE_THRESHOLD`: Lower = accepts more shots
     - `GOAL_PROXIMITY_RADIUS_METERS`: Higher = catches shots from further out

4. **Verify VLM Models**:
   - Ensure Molmo-7B is loaded (check console output)
   - Qwen2.5-VL is optional but improves accuracy

## Troubleshooting

### No Shots Detected?

1. **Check Debug Output**: Look for `[Shot Filter]` messages to see why shots are filtered
2. **Verify Ball Detection**: Ensure ball is being tracked correctly
3. **Check Velocity**: Shots need ball speed >= 8 px/frame
4. **Check Distance**: Shots must be within 120m of goal
5. **Check Angle**: Ball trajectory must be within 150° of goal direction

### Incorrect Classifications?

1. **VLM Quality**: Ensure VLM models are loaded correctly
2. **Frame Quality**: Better video quality = better VLM analysis
3. **Check Prompts**: VLM uses trajectory-based classification (not final outcome)

## Technical Details

### Shot Detection Parameters

```python
SHOT_VELOCITY_THRESHOLD = 8           # px/frame
SHOT_CONFIDENCE_THRESHOLD = 55        # percentage
GOAL_PROXIMITY_RADIUS_METERS = 120    # meters
SHOT_COOLDOWN_FRAMES = 20             # frames
```

### Classification Logic

- **On Target**: Trajectory intersects goal frame rectangle (7.32m × 2.44m)
- **Off Target**: Trajectory misses goal frame (over bar or wide)
- **Blocked**: Defender intercepts before reaching goal
- **Goal**: Ball crosses goal line completely

## Next Steps

If you're still not getting results:

1. Run with `debug=True` and check the console output
2. Look for `[Shot Debug]` messages to see what's being filtered
3. Try lowering thresholds further if needed
4. Check that your video actually contains shots (not just passes)
5. Verify VLM models are working (check for VLM query responses)

## Support

For issues or questions:
- Check console debug output
- Review `shot_detection.py` parameters
- Verify video contains actual shots on goal
- Ensure models are loaded correctly


