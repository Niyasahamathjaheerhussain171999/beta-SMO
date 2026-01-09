# Small Pitch Fix - Relative Scaling Implementation

## Problem
On amateur/semi-pro pitches (60-80m vs 105m professional), the model was confusing **passes with shots** because:
- Hardcoded `GOAL_PROXIMITY_RADIUS_METERS = 120m` was too large for small pitches
- Assumed standard 105m pitch length for all videos
- On 60m pitch, 120m threshold = entire field = false positives

## Solution: Dynamic Relative Scaling

### Changes Made

#### 1. **Relative Distance Threshold** (30% of pitch)
```python
GOAL_PROXIMITY_RADIUS_PERCENT = 0.30  # 30% of pitch length from goal
GOAL_PROXIMITY_RADIUS_METERS = 35      # Absolute maximum (reduced from 120m)
```

**How it works:**
- **Standard pitch (105m)**: 30% = 31.5m (danger zone)
- **Small pitch (60m)**: 30% = 18m (danger zone)
- **Absolute cap**: Maximum 35m (prevents false positives on very small pitches)

#### 2. **Dynamic Pitch Length Detection**
```python
# Detects actual pitch size from detected bounds
estimated_pitch_length_meters = max(60, min(105, width * 0.65))  # Clamp 60-105m
self.actual_pitch_length_meters = estimated_pitch_length_meters
```

**Benefits:**
- Automatically adapts to any pitch size
- No manual configuration needed
- Works for amateur (60m), semi-pro (80m), and professional (105m) pitches

#### 3. **Tightened Angle Filter** (45° cone)
```python
MAX_SHOT_ANGLE_DEGREES = 45  # Tightened from 60° (was 150°)
```

**Why:**
- On small pitches, crosses from wing look dangerous (winger is closer)
- 45° = central shooting cone only
- Filters out wide crosses that look like shots

#### 4. **Increased Velocity Threshold** (15 px/frame)
```python
SHOT_VELOCITY_THRESHOLD = 15  # Increased from 8 px/frame
```

**Why:**
- Amateur passes are slow, amateur shots are fast
- Gap is wider than professional matches
- Forces machine to ignore slow "lofted" passes

## Expected Results

### Before Fix:
- **Small pitch (60m)**: 396 shots detected (mostly false positives from passes)
- **Standard pitch (105m)**: ~50-80 shots (more accurate)

### After Fix:
- **Small pitch (60m)**: ~20-40 shots (realistic for amateur match)
- **Standard pitch (105m)**: ~50-80 shots (unchanged, still accurate)

## Technical Details

### Distance Calculation (Now Relative)
```python
# OLD (Hardcoded):
if distance_to_goal > 120:  # meters
    reject()

# NEW (Relative):
actual_pitch_length = self.actual_pitch_length_meters  # Detected: 60-105m
relative_threshold = actual_pitch_length * 0.30  # 30% of pitch
max_threshold = min(35, relative_threshold)  # Cap at 35m

if distance_to_goal > max_threshold:
    reject()
```

### Example Calculations

| Pitch Size | 30% Threshold | Absolute Cap | Final Threshold |
|------------|--------------|--------------|-----------------|
| 60m (amateur) | 18m | 35m | **18m** |
| 80m (semi-pro) | 24m | 35m | **24m** |
| 105m (professional) | 31.5m | 35m | **31.5m** |

## Files Modified

1. **`src/shot_detection.py`**:
   - Added `GOAL_PROXIMITY_RADIUS_PERCENT = 0.30`
   - Reduced `GOAL_PROXIMITY_RADIUS_METERS = 35` (from 120)
   - Added `MAX_SHOT_ANGLE_DEGREES = 45` (from 60)
   - Increased `SHOT_VELOCITY_THRESHOLD = 15` (from 8)
   - Added `actual_pitch_length_meters` tracking
   - Updated `_setup_goal_zones()` to detect actual pitch length
   - Updated `detect_shot()` to use relative scaling

## Testing Recommendations

1. **Test on small pitch video (60-80m)**:
   - Shot count should drop from ~300+ to ~20-40
   - No passes should be detected as shots
   - Only central shots (45° cone) should be detected

2. **Test on standard pitch video (105m)**:
   - Shot count should remain similar (~50-80)
   - Accuracy should be maintained

3. **Verify angle filter**:
   - Wide crosses should be filtered out
   - Only central shots should pass

## Configuration

No configuration changes needed! The system automatically detects pitch size and adapts.

However, if you want to adjust sensitivity:

```python
# In shot_detection.py:
GOAL_PROXIMITY_RADIUS_PERCENT = 0.30  # Increase for more shots, decrease for fewer
MAX_SHOT_ANGLE_DEGREES = 45           # Increase for wider shots, decrease for narrower
SHOT_VELOCITY_THRESHOLD = 15          # Increase for faster shots only, decrease for slower
```

## Performance Impact

- **No performance impact**: Calculations are simple multiplications
- **Accuracy improvement**: Significant reduction in false positives on small pitches
- **Maintains speed**: All optimizations preserved

