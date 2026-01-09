# 🎯 Shot Classification Fix - All Shots "Off Target" Issue

## Problem Identified

**Issue**: All 18 detected shots were classified as "Shot off target" when manual analysis shows:
- 4 shots on target
- 7 shots off target
- Total: 11 shots (system detected 18)

**Root Cause**: 
1. VLM (Qwen2.5-VL) was responding "SHOT OFF TARGET" for every shot
2. Parsing logic checked "OFF TARGET" first, accepting all VLM responses
3. No angle-based validation to override incorrect VLM classifications

---

## Fixes Applied

### 1. ✅ Improved Video Layer Parsing (`shot_detection.py` line 675-700)

**Before:**
- Checked "OFF TARGET" immediately after "ON TARGET"
- No validation against angle

**After:**
- Checks "ON TARGET" first (before "OFF TARGET")
- Validates "OFF TARGET" responses with angle data
- **Override logic**: If VLM says "OFF TARGET" but angle < 30°, override to "ON TARGET"
- Uses angle-based fallback when VLM response is unclear

**Key Change:**
```python
elif "OFF TARGET" in response_upper:
    # Validate with angle before accepting
    if angle_of_arrival is not None and angle_of_arrival < 30:
        # Narrow angle suggests on target - override VLM
        shot_type = "Shot on target"
        print(f"     ⚠️ VLM said OFF TARGET but angle={angle_of_arrival:.1f}° < 30° → Overriding to ON TARGET")
```

---

### 2. ✅ Enhanced Video Prompt (`shot_detection.py` line 655-658)

**Before:**
- Prompt emphasized "OFF TARGET" classification
- No angle guidance

**After:**
- Added explicit angle guidance in prompt
- Emphasized balanced classification
- Warns against defaulting to "OFF TARGET"
- Includes angle-based suggestion

**Key Addition:**
```
- Trajectory angle {angle_of_arrival:.1f}° suggests: {"ON TARGET" if angle < 45 else "OFF TARGET"}
- With angle {angle_of_arrival:.1f}°, {"classify as ON TARGET" if angle < 45 else "classify as OFF TARGET"} if trajectory is unclear
```

---

### 3. ✅ Improved Static Layer Fallback (`shot_detection.py` line 967-987)

**Before:**
- Defaulted to "off target" for medium angles (30-45°)
- Conservative bias toward "off target"

**After:**
- Defaults to "on target" for medium angles (30-45°) when unclear
- Higher confidence (80%) for narrow angles (<30°)
- More balanced classification

**Key Change:**
```python
# Between 30-45°: Default to ON TARGET when unclear (conservative)
else:
    return "Shot on target", 70  # Was: "Shot off target", 65
```

---

## Expected Results

### Before Fix:
- ❌ All 18 shots = "Shot off target"
- ❌ 0 shots on target
- ❌ VLM bias toward "off target"

### After Fix:
- ✅ Angle-based validation overrides incorrect VLM responses
- ✅ Narrow angles (<30°) → "Shot on target" (even if VLM says "off target")
- ✅ More balanced classification
- ✅ Expected: ~4 shots on target, ~7 shots off target (matching manual)

---

## How It Works Now

### Classification Flow:

1. **VLM Response**: Qwen2.5-VL analyzes video clip
2. **Parse Response**: Check "ON TARGET" first, then "OFF TARGET"
3. **Angle Validation**: 
   - If VLM says "OFF TARGET" but angle < 30° → Override to "ON TARGET"
   - If VLM says "OFF TARGET" and angle > 45° → Accept "OFF TARGET"
   - If VLM unclear → Use angle-based fallback
4. **Final Classification**: Balanced result

### Angle Thresholds:
- **< 30°**: Very likely ON TARGET (high confidence override)
- **30-45°**: Likely ON TARGET (default when unclear)
- **> 45°**: Likely OFF TARGET

---

## Test It

Run analysis again:
```bash
python main4.py --source "input_video/video_segment (1).mp4" --debug
```

**Check output for:**
1. Mix of "Shot on target" and "Shot off target" classifications
2. Override messages: `⚠️ VLM said OFF TARGET but angle=X° < 30° → Overriding to ON TARGET`
3. Shot count closer to 11 (may still be slightly higher due to detection timing)
4. ~4 shots on target, ~7 shots off target

---

## Files Modified

1. **shot_detection.py**:
   - Line 655-658: Enhanced video prompt with angle guidance
   - Line 675-700: Improved parsing with angle validation and override logic
   - Line 967-987: Better static layer fallback (defaults to "on target" for medium angles)

---

## Summary

The fix addresses the VLM bias by:
- ✅ Validating VLM responses against angle data
- ✅ Overriding incorrect "OFF TARGET" classifications when angle suggests "ON TARGET"
- ✅ More balanced prompts that don't default to "OFF TARGET"
- ✅ Better fallback logic that defaults to "ON TARGET" when unclear

**Expected**: Balanced classification matching manual analysis (4 on target, 7 off target).


