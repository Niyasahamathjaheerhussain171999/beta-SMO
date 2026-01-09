# 🎯 Shot Detection System - Complete Explanation

## Table of Contents
1. [Why Shots Aren't Detected](#why-shots-arent-detected)
2. [Shot Detection Pipeline](#shot-detection-pipeline)
3. [Metrics Explained](#metrics-explained)
4. [Troubleshooting Guide](#troubleshooting-guide)

---

## Why Shots Aren't Detected

### Common Reasons & Exact Fixes

#### ❌ **Reason 1: Ball Not Detected**
**Problem**: The ball detection model (`football-ball-detection.pt`) doesn't see the ball in the frame.

**How to Check**:
- Look at console output: `Ball Detection Model: 820 inferences`
- Check if `ball_xy` is `None` in logs
- Ball might be too small, blurry, or occluded

**Fix**:
- Improve video quality (higher resolution)
- Check ball detection model confidence threshold
- Ensure ball is visible in video frames

**Code Location**: `main4.py` line 1145-1163

---

#### ❌ **Reason 2: Ball Speed Too Low**
**Problem**: Ball velocity is below the minimum threshold.

**Current Threshold**: `SHOT_VELOCITY_THRESHOLD = 8` pixels/frame

**What This Means**:
- At 25 FPS: 8 px/frame = 200 pixels/second
- At 54 FPS (your video): 8 px/frame = 432 pixels/second
- Slow passes or rolling balls won't trigger

**How to Check**:
```python
# In shot_detection.py, line 442
speed, dx, dy = self.get_ball_velocity()
if speed < 8:  # Filtered out here
```

**Fix**:
- Lower threshold: Change `SHOT_VELOCITY_THRESHOLD = 8` to `SHOT_VELOCITY_THRESHOLD = 5` in `shot_detection.py` line 40
- **Warning**: Lower values may detect passes as shots

**Code Location**: `shot_detection.py` line 40, 456-459

---

#### ❌ **Reason 3: Distance Too Far from Goal**
**Problem**: Ball is more than 120 meters from the goal.

**Current Threshold**: `GOAL_PROXIMITY_RADIUS_METERS = 120` meters

**What This Means**:
- Standard pitch length: ~105 meters
- 120m covers entire pitch + some margin
- Shots from own half might be filtered

**How to Check**:
```python
# In shot_detection.py, line 445
distance_to_goal, target_goal = self.calculate_distance_to_goal(ball_xy)
if distance_to_goal > 120:  # Filtered out here
```

**Fix**:
- Increase radius: Change `GOAL_PROXIMITY_RADIUS_METERS = 120` to `GOAL_PROXIMITY_RADIUS_METERS = 150` in `shot_detection.py` line 47
- **Warning**: Too high may detect long passes as shots

**Code Location**: `shot_detection.py` line 47, 468-471

---

#### ❌ **Reason 4: Trajectory Angle Too Wide**
**Problem**: Ball trajectory doesn't point toward goal (angle > 150°).

**Current Threshold**: Maximum angle deviation = **150 degrees**

**What This Means**:
- 0° = Directly toward goal center
- 90° = Perpendicular to goal
- 150° = Very wide angle (almost backward)
- Shots at extreme angles are filtered

**How to Check**:
```python
# In shot_detection.py, line 474-481
vector_intersects, angle_of_arrival = self.check_velocity_vector_intersection(...)
if not vector_intersects:  # Filtered out here
```

**Fix**:
- Increase angle tolerance: Change line 324 in `shot_detection.py` from `if angle_degrees > 150:` to `if angle_degrees > 170:`
- **Warning**: Too wide may detect crosses as shots

**Code Location**: `shot_detection.py` line 324, 478-481

---

#### ❌ **Reason 5: Player Not in Shooting Range**
**Problem**: Shooter position doesn't meet shooting range criteria.

**Current Range**: 
- Width: 5% to 95% of frame (central 90%)
- Height: 2.5% to 97.5% of frame (central 95%)

**What This Means**:
- Excludes extreme corners where crosses occur
- Very permissive (almost entire pitch)

**How to Check**:
```python
# In shot_detection.py, line 448
in_shooting_range = self.is_in_shooting_range(shooter_pos)
if not in_shooting_range:  # Filtered out here
```

**Fix**:
- Usually not the issue (range is very wide)
- If needed, adjust `is_in_shooting_range()` in `shot_detection.py` line 371-396

**Code Location**: `shot_detection.py` line 371-396, 462-465

---

#### ❌ **Reason 6: VLM Says "Not a Shot"**
**Problem**: AI model (Molmo-7B or Qwen2.5-VL) classifies action as pass/cross, not shot.

**Current Confidence Threshold**: `SHOT_CONFIDENCE_THRESHOLD = 55%`

**What This Means**:
- VLM analyzes frame and determines if it's a shot
- If confidence < 55%, shot is rejected
- VLM might see it as a pass, cross, or clearance

**How to Check**:
- Look for console messages: `[Shot Filtered] Frame X: VLM said NOT a shot`
- Check VLM response in logs

**Fix**:
- Lower confidence: Change `SHOT_CONFIDENCE_THRESHOLD = 55` to `SHOT_CONFIDENCE_THRESHOLD = 45` in `shot_detection.py` line 44
- Improve VLM prompts (already optimized)
- **Warning**: Lower values may create false positives

**Code Location**: `shot_detection.py` line 44, 552-570

---

#### ❌ **Reason 7: Cooldown Period Active**
**Problem**: Shot detected too recently (within last 20 frames).

**Current Cooldown**: `SHOT_COOLDOWN_FRAMES = 20` frames

**What This Means**:
- At 25 FPS: 20 frames = 0.8 seconds
- At 54 FPS: 20 frames = 0.37 seconds
- Prevents duplicate detections of same shot

**How to Check**:
```python
# In shot_detection.py, line 438
if frame_idx - self.last_shot_frame < 20:  # Filtered out here
```

**Fix**:
- Reduce cooldown: Change `SHOT_COOLDOWN_FRAMES = 20` to `SHOT_COOLDOWN_FRAMES = 10` in `shot_detection.py` line 42
- **Warning**: Too low may create duplicate detections

**Code Location**: `shot_detection.py` line 42, 438-439

---

## Shot Detection Pipeline

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO FRAME INPUT                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: BALL DETECTION                                     │
│  • YOLO model detects ball in frame                         │
│  • Returns: ball coordinates (x, y) or None                │
│  Location: main4.py:1145                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────┴─────────────┐
         │  Is ball detected?        │
         └─────────────┬─────────────┘
                       │
            ┌──────────┴──────────┐
            │ NO                  │ YES
            │                     │
            ▼                     ▼
    [STOP: No ball]      ┌─────────────────────┐
                         │ STEP 2: BALL TRACKING│
                         │ • Add to ball_history│
                         │ • Calculate velocity │
                         │ Location: main4.py:  │
                         │ 1161-1168            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ STEP 3: FIND CLOSEST PLAYER    │
                    │ • Calculate distance to players│
                    │ • Find shooter (closest player)│
                    │ Location: main4.py:1446-1467   │
                    └──────────┬──────────────────────┘
                               │
                               ▼
                    ┌───────────────────────────────┐
                    │ STEP 4: GEOMETRIC FILTERS     │
                    │ (All must pass)               │
                    └──────────┬──────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Filter 1:     │    │ Filter 2:     │    │ Filter 3:     │
│ Ball Speed    │    │ Distance      │    │ Angle         │
│ ≥ 8 px/frame  │    │ ≤ 120m       │    │ ≤ 150°       │
│               │    │               │    │               │
│ Location:     │    │ Location:     │    │ Location:     │
│ line 456      │    │ line 468      │    │ line 478      │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────┴────────┐
                    │ All passed?     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ NO              │ YES
                    │                 │
                    ▼                 ▼
            [STOP: Filtered]  ┌──────────────────────┐
                               │ STEP 5: AI ANALYSIS  │
                               │ (VLM Stages)        │
                               └──────────┬───────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │                               │
                          ▼                               ▼
                ┌──────────────────┐          ┌──────────────────┐
                │ Stage 1:         │          │ Stage 2:          │
                │ Is this a shot?  │          │ Classify outcome  │
                │ (vs pass/cross)  │          │ (on/off target)   │
                │                  │          │                    │
                │ Location:        │          │ Location:          │
                │ line 617-668     │          │ line 670-801      │
                └──────────┬───────┘          └──────────┬─────────┘
                           │                            │
                    ┌──────┴──────┐              ┌──────┴──────┐
                    │ NO          │ YES          │ Result:     │
                    │             │              │ Shot type   │
                    ▼             ▼              │ + Confidence│
            [STOP: Not shot]  ┌──────────────────┴─────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ STEP 6: FINAL CHECK  │
                    │ Confidence ≥ 55%?    │
                    │ Location: line 552   │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │ NO                 │ YES
                    │                    │
                    ▼                    ▼
            [STOP: Low conf]    ┌──────────────────┐
                                │ SHOT DETECTED!   │
                                │ Record event     │
                                │ Location:        │
                                │ main4.py:1488    │
                                └──────────────────┘
```

---

### Step-by-Step Explanation

#### **STEP 1: Ball Detection** (`main4.py:1145`)
```python
b_det = sv.Detections.from_ultralytics(b_m(frame, imgsz=640, verbose=False)[0])
```
- Uses YOLO ball detection model
- Input: Video frame (640px width)
- Output: Ball detections with coordinates
- **If no ball detected**: Process stops here

---

#### **STEP 2: Ball Tracking** (`main4.py:1161-1168`)
```python
ball_coords = b_det.get_anchors_coordinates(sv.Position.CENTER)
ball_xy = ball_coords[0] if len(ball_coords) > 0 else None
if ball_xy is not None:
    ball_history.append((f_idx, ball_xy.copy()))
```
- Extracts ball center coordinates
- Adds to history buffer (last 10 positions)
- Used to calculate velocity

---

#### **STEP 3: Find Closest Player** (`main4.py:1446-1467`)
```python
for tid, p_xy, bbox in zip(p_det.tracker_id, ...):
    dist = np.linalg.norm(ball_xy - p_xy)
    if dist < min_dist and dist < BALL_PROXIMITY_THRESHOLD * 1.5:
        shooter_tid = tid
        shooter_pos = p_xy
```
- Finds player closest to ball
- Must be within 97.5px (65px * 1.5) for shot detection
- Identifies potential shooter

---

#### **STEP 4: Geometric Filters** (`shot_detection.py:454-481`)

**Filter 1: Ball Speed** (Line 456)
```python
if speed < SHOT_VELOCITY_THRESHOLD:  # 8 px/frame
    return False, None, 0, None
```
- Calculates velocity from ball history
- Requires minimum speed of 8 pixels/frame
- Filters out slow passes/rolls

**Filter 2: Distance to Goal** (Line 468)
```python
if distance_to_goal > GOAL_PROXIMITY_RADIUS_METERS:  # 120m
    return False, None, 0, None
```
- Calculates distance from ball to nearest goal
- Uses pitch calibration (pixels to meters)
- Filters shots from too far away

**Filter 3: Trajectory Angle** (Line 478)
```python
vector_intersects, angle_of_arrival = self.check_velocity_vector_intersection(...)
if not vector_intersects:  # Angle > 150°
    return False, None, 0, None
```
- Checks if ball velocity vector points toward goal
- Maximum angle deviation: 150 degrees
- Filters sideways/backward movements

---

#### **STEP 5: AI Analysis** (`shot_detection.py:583-600`)

**Stage 1: Shot vs Pass** (Line 617-668)
```python
is_shot, shot_conf = self._vlm_stage1_is_shot(frame_crop, ball_speed, distance_to_goal)
```
- VLM analyzes frame crop (500x500px around ball)
- Determines: Shot vs Pass/Cross/Clearance
- Returns: `(True/False, confidence)`

**Stage 2: Shot Classification** (Line 670-801)
```python
shot_type, outcome_conf = self._vlm_stage2_shot_outcome(frame_crop, angle_of_arrival)
```
- If Stage 1 confirms shot, classifies outcome:
  - **"Shot on target"**: Trajectory toward goal frame
  - **"Shot off target"**: Trajectory over/wide of goal
  - **"Shot blocked"**: Defender intercepts
  - **"Goal"**: Ball crosses goal line

---

#### **STEP 6: Final Confidence Check** (Line 552)
```python
if is_shot and vlm_conf >= SHOT_CONFIDENCE_THRESHOLD:  # 55%
    return True, shot_type, vlm_conf, metrics
```
- Combines Stage 1 (40%) and Stage 2 (60%) confidence
- Final threshold: 55%
- Records shot event if passed

---

## Metrics Explained

### Ball Velocity (`ball_velocity_px_per_frame`)

**What it is**: Speed of ball movement in pixels per frame

**How calculated**:
```python
# shot_detection.py:213-234
# Average velocity over last 3 ball positions
speed = sqrt(dx² + dy²)
```

**Example values**:
- **Slow pass**: 3-5 px/frame
- **Normal shot**: 10-20 px/frame
- **Power shot**: 25-40 px/frame

**Threshold**: `SHOT_VELOCITY_THRESHOLD = 8` px/frame

**At different FPS**:
- 25 FPS: 8 px/frame = 200 px/second
- 30 FPS: 8 px/frame = 240 px/second
- 54 FPS: 8 px/frame = 432 px/second

---

### Distance to Goal (`distance_from_goal_meters`)

**What it is**: Straight-line distance from ball to nearest goal center

**How calculated**:
```python
# shot_detection.py:236-273
# Euclidean distance: sqrt((ball_x - goal_x)² + (ball_y - goal_y)²)
# Converted to meters using pixels_per_meter calibration
```

**Example values**:
- **Penalty box**: 10-18 meters
- **Edge of box**: 18-20 meters
- **Midfield**: 50-60 meters
- **Own half**: 60-100 meters

**Threshold**: `GOAL_PROXIMITY_RADIUS_METERS = 120` meters

**Calibration**:
- Standard pitch: 105m long
- `pixels_per_meter = frame_width / 105`

---

### Angle of Arrival (`angle_of_arrival_degrees`)

**What it is**: Angle between ball velocity vector and direction to goal

**How calculated**:
```python
# shot_detection.py:275-327
# Angle = arccos(dot(velocity_unit, goal_direction_unit))
angle_degrees = degrees(angle_rad)
```

**Example values**:
- **0-30°**: Directly toward goal (likely on target)
- **30-50°**: Slight angle (still likely on target)
- **50-90°**: Wide angle (likely off target)
- **>150°**: Almost backward (filtered out)

**Threshold**: Maximum 150 degrees deviation

**Visual**:
```
     Goal
      │
      │ 0° (direct)
      │
      ├─ 30° (slight angle)
      │
      ├─ 90° (perpendicular)
      │
      └─ 150° (very wide - filtered)
```

---

### VLM Confidence Score (`vlm_confidence_score`)

**What it is**: AI model's confidence that action is a shot

**How calculated**:
```python
# shot_detection.py:597-599
# Weighted combination:
final_conf = (stage1_conf * 0.4) + (stage2_conf * 0.6)
```

**Stage 1 Confidence** (Shot vs Pass):
- **88%**: VLM says "YES" decisively
- **30%**: VLM says "NO" or uncertain

**Stage 2 Confidence** (Shot Type):
- **92%**: Clear goal
- **88%**: Clear on/off target
- **85%**: Blocked shot
- **65-72%**: Uncertain (fallback)

**Final Threshold**: `SHOT_CONFIDENCE_THRESHOLD = 55%`

**Example**:
- Stage 1: 88% (shot confirmed)
- Stage 2: 88% (on target)
- Final: (88 * 0.4) + (88 * 0.6) = **88%** ✅

---

### Velocity Components (`velocity_dx`, `velocity_dy`)

**What they are**: Ball movement in X and Y directions

**dx (horizontal)**:
- **Positive**: Moving right (toward right goal)
- **Negative**: Moving left (toward left goal)
- **Large value**: Fast horizontal movement

**dy (vertical)**:
- **Positive**: Moving down (toward bottom of frame)
- **Negative**: Moving up (toward top of frame)
- **Large value**: Fast vertical movement

**Used for**: Trajectory calculation and angle determination

---

## Troubleshooting Guide

### Issue: No Shots Detected at All

**Checklist**:

1. **Ball Detection Working?**
   ```python
   # Add debug: main4.py:1145
   print(f"Frame {f_idx}: Ball detections = {len(b_det)}")
   ```
   - If always 0: Ball model not detecting ball
   - **Fix**: Check video quality, ball visibility

2. **Ball Tracking Working?**
   ```python
   # Check: main4.py:1163
   print(f"Frame {f_idx}: ball_xy = {ball_xy}")
   ```
   - If always None: Ball detected but not tracked
   - **Fix**: Check ball detection confidence

3. **Ball Speed High Enough?**
   ```python
   # Check: shot_detection.py:442
   speed, dx, dy = self.get_ball_velocity()
   print(f"Speed: {speed} px/frame (threshold: 8)")
   ```
   - If speed < 8: Ball moving too slowly
   - **Fix**: Lower `SHOT_VELOCITY_THRESHOLD` to 5

4. **Distance Reasonable?**
   ```python
   # Check: shot_detection.py:445
   distance_to_goal, target_goal = self.calculate_distance_to_goal(ball_xy)
   print(f"Distance: {distance_to_goal}m (threshold: 120m)")
   ```
   - If distance > 120: Too far from goal
   - **Fix**: Increase `GOAL_PROXIMITY_RADIUS_METERS` to 150

5. **Angle Acceptable?**
   ```python
   # Check: shot_detection.py:474
   vector_intersects, angle = self.check_velocity_vector_intersection(...)
   print(f"Angle: {angle}° (max: 150°)")
   ```
   - If angle > 150: Trajectory too wide
   - **Fix**: Increase angle tolerance to 170°

6. **VLM Confidence?**
   - Look for: `[Shot Filtered] VLM said NOT a shot`
   - **Fix**: Lower `SHOT_CONFIDENCE_THRESHOLD` to 45

---

### Issue: Shots Detected But Wrong Type (On vs Off Target)

**Checklist**:

1. **VLM Stage 2 Classification**
   - Check VLM response in logs
   - May misclassify trajectory
   - **Fix**: Improve VLM prompts (already optimized)

2. **Angle of Arrival**
   - Wide angles (>50°) often off target
   - Narrow angles (<30°) often on target
   - **Fix**: Adjust angle-based fallback logic

---

### Issue: Too Many False Positives (Passes Detected as Shots)

**Fixes**:
1. **Increase velocity threshold**: `SHOT_VELOCITY_THRESHOLD = 10`
2. **Decrease distance radius**: `GOAL_PROXIMITY_RADIUS_METERS = 100`
3. **Decrease angle tolerance**: Change 150° to 120°
4. **Increase confidence**: `SHOT_CONFIDENCE_THRESHOLD = 65`

---

### Issue: Shots Missed (Should Detect But Don't)

**Fixes**:
1. **Decrease velocity threshold**: `SHOT_VELOCITY_THRESHOLD = 5`
2. **Increase distance radius**: `GOAL_PROXIMITY_RADIUS_METERS = 150`
3. **Increase angle tolerance**: Change 150° to 170°
4. **Decrease confidence**: `SHOT_CONFIDENCE_THRESHOLD = 45`

---

## Quick Reference: All Thresholds

| Parameter | Current Value | Location | What It Does |
|-----------|--------------|----------|--------------|
| `SHOT_VELOCITY_THRESHOLD` | 8 px/frame | `shot_detection.py:40` | Minimum ball speed |
| `GOAL_PROXIMITY_RADIUS_METERS` | 120 meters | `shot_detection.py:47` | Maximum distance from goal |
| `SHOT_CONFIDENCE_THRESHOLD` | 55% | `shot_detection.py:44` | Minimum VLM confidence |
| `SHOT_COOLDOWN_FRAMES` | 20 frames | `shot_detection.py:42` | Time between shots |
| `Angle Tolerance` | 150 degrees | `shot_detection.py:324` | Maximum trajectory angle |
| `BALL_PROXIMITY_THRESHOLD` | 65 pixels | `main4.py` (global) | Max distance for ownership |

---

## Summary

**Shot detection requires**:
1. ✅ Ball detected and tracked
2. ✅ Ball speed ≥ 8 px/frame
3. ✅ Distance ≤ 120m from goal
4. ✅ Trajectory angle ≤ 150°
5. ✅ VLM confidence ≥ 55%
6. ✅ Player in shooting range
7. ✅ Not in cooldown period

**If shots aren't detected**, check each filter in order and adjust thresholds accordingly.


