# Shot Detection Fixes - Summary

## Issues Fixed

### 1. ✅ Code Error Fixed
**Error**: `ValueError: The truth value of an array with more than one element is ambiguous`

**Location**: `main4.py` line 1532

**Fix**: Changed `if ball_xy and shooter_pos is not None:` to `if ball_xy is not None and shooter_pos is not None:`

**Why**: `ball_xy` is a numpy array, and Python can't evaluate numpy arrays directly in boolean context.

---

### 2. ✅ Referee Shots Filtered Out
**Problem**: Shots were being attributed to referees (e.g., "Referee #150", "Referee #242")

**Fix**: Added referee filtering in shot detection (same as pass detection)

**Location**: `main4.py` line 1488-1490

**Code**:
```python
# Skip referee shots (referees don't take shots)
if shooter_team == "Referee":
    continue
```

---

### 3. ✅ Improved On/Off Target Classification
**Problem**: All shots were classified as "Shot on target" (30 detected, all on target)
**Expected**: 4 on target, 7 off target

**Fixes Applied**:

#### a) Improved VLM Video Prompt (`shot_detection.py` line 504-530)
- Added explicit distinction between ON TARGET and OFF TARGET
- Emphasized trajectory analysis
- Added angle-based guidance (< 45° = on target, > 45° = off target)

#### b) Improved Response Parsing (`shot_detection.py` line 527-545)
- **Changed priority**: Check OFF TARGET patterns FIRST (more specific)
- OFF TARGET patterns checked before ON TARGET
- Better keyword matching for "over bar", "wide", "miss"

#### c) Enhanced Angle-Based Fallback (`shot_detection.py` line 970-990)
- If angle > 45°: Default to "Shot off target"
- If angle < 30°: Default to "Shot on target"
- Between 30-45°: Use trajectory keywords in response

**Key Change**: OFF TARGET patterns are now checked BEFORE ON TARGET patterns, preventing false "on target" classifications.

---

### 4. ✅ Reduced False Positives
**Problem**: 30 shots detected when only 11 should be detected

**Fixes Applied**:

#### a) Increased Cooldown (`shot_detection.py` line 56)
- **Before**: `SHOT_COOLDOWN_FRAMES = 20` (0.8 seconds at 25 FPS)
- **After**: `SHOT_COOLDOWN_FRAMES = 30` (1.2 seconds at 25 FPS)
- **Effect**: Prevents duplicate detections of same shot

#### b) Increased Confidence Threshold (`shot_detection.py` line 64)
- **Before**: `SHOT_CONFIDENCE_THRESHOLD = 55%`
- **After**: `SHOT_CONFIDENCE_THRESHOLD = 60%`
- **Effect**: Only accepts higher-confidence shots, reduces false positives

---

## Expected Results After Fixes

### Before Fixes:
- ❌ 30 shots detected (too many)
- ❌ All classified as "on target" (wrong)
- ❌ Many attributed to referees (wrong)
- ❌ Code crash at end

### After Fixes:
- ✅ Fewer false positives (cooldown + confidence threshold)
- ✅ Better on/off target classification (improved VLM prompts + parsing)
- ✅ No referee shots (filtered out)
- ✅ No code crashes (numpy array fix)

---

## How to Verify

Run the analysis again:
```bash
python main4.py --source "input_video/video_segment (1).mp4" --debug
```

**Check**:
1. Shot count should be closer to 11 (may still be slightly higher due to detection timing)
2. Should see both "Shot on target" and "Shot off target" classifications
3. No shots attributed to "Referee"
4. No code crashes

---

## If Still Not Accurate

### Too Many Shots Detected?
1. **Increase cooldown**: Change `SHOT_COOLDOWN_FRAMES = 30` to `40` or `50`
2. **Increase confidence**: Change `SHOT_CONFIDENCE_THRESHOLD = 60` to `65` or `70`
3. **Increase velocity threshold**: Change `SHOT_VELOCITY_THRESHOLD = 8` to `10` or `12`

### Still All "On Target"?
1. Check VLM responses in console - look for "OFF TARGET" keywords
2. Check angle values - shots with angle > 45° should be off target
3. Verify video quality - VLM needs clear view of ball trajectory

### Still Detecting Referees?
- Check team detection - ensure referees are properly identified
- The filter should catch them, but if team detection is wrong, shots may still be attributed incorrectly

---

## Technical Details

### Classification Logic Flow:
1. VLM analyzes video clip/frame
2. Response parsed for keywords:
   - **OFF TARGET** checked FIRST (more specific patterns)
   - **ON TARGET** checked second
   - **GOAL** checked first (highest priority)
   - **BLOCKED** checked second
3. If no clear match, use angle-based fallback:
   - Angle > 45° → "Shot off target"
   - Angle < 30° → "Shot on target"
   - 30-45° → Use trajectory keywords

### Threshold Changes:
- **Cooldown**: 20 → 30 frames (50% increase)
- **Confidence**: 55% → 60% (9% increase)
- **Angle threshold**: 45° (for off target classification)

---

## Files Modified

1. `main4.py`:
   - Line 1488-1490: Added referee filtering
   - Line 1532: Fixed numpy array boolean check

2. `shot_detection.py`:
   - Line 56: Increased cooldown to 30 frames
   - Line 64: Increased confidence threshold to 60%
   - Line 504-530: Improved video prompt for better on/off target distinction
   - Line 527-545: Improved response parsing (OFF TARGET first)
   - Line 970-990: Enhanced angle-based fallback logic

---

## Next Steps

1. **Run analysis** and check results
2. **Compare** detected shots vs manual analysis
3. **Adjust thresholds** if needed (see "If Still Not Accurate" above)
4. **Review VLM responses** in debug output to see what AI is seeing


