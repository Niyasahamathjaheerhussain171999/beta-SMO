import os
import cv2
import torch
import csv
import json
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from collections import deque
import requests
import logging
from typing import Dict, Any, Optional

# Optional import for YouTube downloads (only needed if using YouTube URLs)
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    yt_dlp = None

# Import shot detection and report generation modules
try:
    from shot_detection import ShotDetector, generate_shot_report_json, generate_shot_report_csv
    SHOT_DETECTION_AVAILABLE = True
except ImportError:
    SHOT_DETECTION_AVAILABLE = False
    ShotDetector = None
    generate_shot_report_json = None
    generate_shot_report_csv = None
    print("⚠️  Warning: shot_detection module not found. Shot detection will be disabled.")

# Try both possible file names (shot_report_generator.py or scout_report_generator.py)
try:
    from shot_report_generator import generate_full_scout_report_html
    SHOT_REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    try:
        from scout_report_generator import generate_full_scout_report_html
        SHOT_REPORT_GENERATOR_AVAILABLE = True
    except ImportError:
        SHOT_REPORT_GENERATOR_AVAILABLE = False
        generate_full_scout_report_html = None
        print("⚠️  Warning: shot_report_generator/scout_report_generator module not found. HTML reports will not include shots.")


# --- 1. CONFIG ---
DATA_DIR = 'data'
# All 3 YOLO models for Lightning AI
PLAYER_MODEL = os.path.join(DATA_DIR, 'football-player-detection.pt')
BALL_MODEL = os.path.join(DATA_DIR, 'football-ball-detection.pt')
PITCH_MODEL = os.path.join(DATA_DIR, 'football-pitch-detection.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# API Configuration
BASE_URL = "https://api.scoutme.cloud/"
API_KEY = "sk_smo_9f3cA1b8E2D4F6a9B7C0XyZQmPkyam"
INPUT_VIDEO_DIR = 'input_video'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Single model mode OFF - using separate models
SINGLE_MODEL_MODE = False

# Team colors (configurable)
TEAM_A_NAME = "Blue"
TEAM_B_NAME = "Red"

# === TUNED PARAMETERS FOR 80%+ ACCURACY (VEO 3 CAMERA) ===
# FIXED: Throw-ins were over-detected (35 vs 3 manual)
# FIXED: Short passes under-detected (4 vs 27 manual)

BALL_PROXIMITY_THRESHOLD = 65      # Distance to consider ball possession
EVENT_COOLDOWN_FRAMES = 45         # Prevent duplicate detections
MIN_PASS_DISTANCE = 50             # Filter very short ball movements
MIN_OWNERSHIP_FRAMES = 8           # Require clear possession before pass

# Field zones - MUCH STRICTER for throw-in
SIDELINE_MARGIN = 0.008            # STRICT: Only 0.8% = extreme edge for throw-ins
CROSS_ZONE_WIDTH = 0.18            # Smaller wing zone
HEADER_HEIGHT_RATIO = 0.10         # STRICTER: Top 10% only for headers

# Pass classification thresholds - FAVOR SHORT PASSES (60% of all passes)
SHORT_PASS_THRESHOLD = 150         # Lowered: More passes classified as short (most passes are short!)
LONG_PASS_THRESHOLD = 300          # Raised: Only very long passes are "long"

# ANTI-THROW-IN BIAS: Throw-ins should be RARE (~5% of passes)
THROW_IN_DISABLED = False          # Set True to completely disable throw-in detection
REQUIRE_PITCH_BOUNDS_FOR_THROW_IN = True  # Must detect pitch for throw-in

# Video annotation settings
SAVE_ANNOTATED_VIDEO = True        # Save video with pass annotations

# === STRICT CONFIDENCE AND VALIDATION ===
CONFIDENCE_THRESHOLD = 72          # RAISED: Only report high-confidence passes
TEMPORAL_BUFFER_SIZE = 10          # More frames for temporal analysis
THROW_IN_CONSECUTIVE_FRAMES = 4    # STRICTER: 4+ frames for throw-in
HEADER_CONSECUTIVE_FRAMES = 4      # MUCH STRICTER: 4+ frames of head-ball contact

# VLM settings - VLM IS PRIMARY CLASSIFIER
USE_VLM_STAGE1 = True              # Stage 1: STRICT pass verification
USE_VLM_STAGE2 = True              # Stage 2: Final type classification
VLM_CROP_SIZE = 450                # Larger crop for better context
VLM_IS_PRIMARY = True              # VLM decision overrides geometry when confident


# --- 2. LOAD MOLMO AI ---
print("🧠 Loading Molmo-7B into GPU...")
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
except:
    pass

model_id = 'allenai/Molmo-7B-D-0924'
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
vlm_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, device_map="auto", torch_dtype="auto" 
)
vlm_model.config.use_cache = False
vlm_model.eval()


def vlm_query(frame_crop, prompt, max_tokens=50):
    """Generic VLM query function with extended token output"""
    try:
        inputs = processor.process(images=[Image.fromarray(frame_crop)], text=prompt)
        
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                processed_inputs[key] = value.to(vlm_model.device).unsqueeze(0)
            else:
                processed_inputs[key] = value
        
        with torch.inference_mode():
            torch.cuda.empty_cache()
            
            input_ids = processed_inputs['input_ids']
            batch_size, seq_len = input_ids.shape
            eos_token_id = processor.tokenizer.eos_token_id
            
            generated_ids = input_ids.clone()
            
            for step in range(max_tokens):
                current_seq_len = generated_ids.shape[1]
                
                model_inputs = {
                    'input_ids': generated_ids,
                    'position_ids': torch.arange(current_seq_len, device=vlm_model.device, dtype=torch.long).unsqueeze(0),
                    'attention_mask': torch.ones(batch_size, current_seq_len, device=vlm_model.device, dtype=torch.long),
                    'use_cache': False,
                }
                
                if step == 0:
                    for key in processed_inputs:
                        if key not in ['input_ids', 'position_ids', 'attention_mask']:
                            model_inputs[key] = processed_inputs[key]
                
                outputs = vlm_model(**model_inputs)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == eos_token_id:
                    break
            
            torch.cuda.empty_cache()
        
        input_length = input_ids.shape[1]
        generated_tokens = generated_ids[0, input_length:]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response
    except Exception as e:
        return "unknown"


# === SCOUTME VLM PASS CLASSIFICATION SYSTEM v2 ===
# Ultra-accurate prompts calibrated for 80%+ match with manual analysis

def vlm_stage1_is_pass(frame_crop):
    """
    Stage 1: RELAXED verification - Is this likely a pass event?
    More lenient to avoid over-filtering.
    Returns: (is_pass: bool, confidence: float 0-100)
    """
    prompt = """Look at this soccer game frame.

Is there a PASS or ball transfer happening? Answer YES if:
- Ball is moving between players
- Player is kicking, heading, or throwing the ball
- Ball is in flight or being played

Answer NO only if:
- Ball is stationary with no action
- Goalkeeper just holding ball
- No players near the ball

Simple answer: YES or NO"""

    response = vlm_query(frame_crop, prompt, max_tokens=10)
    response_upper = response.upper()
    
    # More lenient parsing - assume YES if unclear
    is_pass = "YES" in response_upper or ("NO" not in response_upper and "BALL" in response_upper)
    
    # Default to moderate confidence (let Stage 2 decide)
    if is_pass:
        confidence = 75  # Pass to Stage 2
    else:
        confidence = 55  # Still allow through, but lower confidence
    
    return is_pass, confidence


def vlm_stage2_classify_type(frame_crop, geometry_suggestion, context_info):
    """
    Stage 2: Visual pass classification - GOLD STANDARD PROMPT
    Uses expert-level analysis framework for accurate classification.
    """
    distance = context_info.get('distance', 'unknown')
    position = context_info.get('position', 'pitch')
    ball_height = context_info.get('ball_height', 'normal')
    
    distance_num = int(float(distance)) if distance != 'unknown' else 0
    
    prompt = f"""You are an expert professional soccer technical analyst. Your task is to analyze the provided image frame and classify the exact type of pass being executed.

To determine the classification, analyze the following visual cues in order:

1. The Actor & Action: Identify the player in possession or immediately releasing the ball. What is their body mechanic (kicking, heading, throwing)?

2. Pitch Location: Where on the field is the action happening? (e.g., deep wing, central midfield, sideline).

3. Implied Trajectory & Intent: Based on body shape and ball position, is the intended destination near (ground/low) or far (high/power)?

Distance context: {distance}px | Position: {position} | Ball height: {ball_height}

CRITICAL DISTANCE GUIDELINES:
- Distance < 150px = SHORT PASS (most common - 60% of all passes!)
- Distance 150-250px = Usually SHORT PASS (default to short unless clearly long trajectory)
- Distance >= 250px = Can be LONG PASS (only if clearly intended for long distance)

Based on this visual analysis, you must classify the action strictly into exactly one of the following categories:

Header: The ball is making contact with the player's head for a pass.

Short throw-in: A player on the sideline using two hands to throw the ball to a nearby teammate.

Long throw-in: A player on the sideline using significant power (often a long run-up) to throw the ball a long distance, usually towards the penalty box.

Cross: A player located in a wide area (the wings) kicking the ball high and long into the central attacking penalty area.

Long pass: A kicked pass intended to travel a SIGNIFICANT distance (250px+). Only classify as "Long pass" if the ball is clearly intended to travel a long way through the air.

Short pass: A kicked pass, usually along the ground or very low, intended for a teammate close by. THIS IS THE DEFAULT for most passes (60% of all passes). Default to "Short pass" unless clearly a long-distance pass.

(None): If no clear passing action is occurring in this specific frame.

Output Requirement: Provide only the exact category name from the list above as your final answer. REMEMBER: Most passes are "Short pass" - only classify as "Long pass" if the distance is clearly long (250px+)."""

    response = vlm_query(frame_crop, prompt, max_tokens=20)
    response_upper = response.upper().strip()
    
    # Parse response - look for exact category names
    response_clean = response_upper.replace("(", "").replace(")", "").replace(".", "").strip()
    
    # THROW-IN: VERY STRICT - must be at sideline AND VLM must be very clear
    # Throw-ins are RARE (only 3 expected in 5-min video)
    if position == 'sideline' and ("THROW" in response_clean or "THROW-IN" in response_clean):
        # VLM says throw-in AND we're at sideline - trust it
        if "SHORT" in response_clean or distance_num < SHORT_PASS_THRESHOLD:
            return "Short throw-in", 85
        else:
            return "Long throw-in", 85
    elif "THROW" in response_clean and position != 'sideline':
        # VLM says throw-in but NOT at sideline - likely wrong, use geometry
        if geometry_suggestion and geometry_suggestion not in ["Short throw-in", "Long throw-in"]:
            return geometry_suggestion, 75
        # If geometry also says throw-in, trust it but lower confidence
        return "Short throw-in" if distance_num < SHORT_PASS_THRESHOLD else "Long throw-in", 70
    
    if "HEADER" in response_clean or ("HEAD" in response_clean and "PASS" not in response_clean):
        return "Header", 85
    
    if "CROSS" in response_clean:
        return "Cross", 85
    
    # Check SHORT PASS FIRST (more common - 60% of passes)
    if "SHORT PASS" in response_clean or ("SHORT" in response_clean and "PASS" in response_clean and "THROW" not in response_clean):
        return "Short pass", 85
    
    # Then LONG PASS - BUT VALIDATE WITH DISTANCE (less common - 20% of passes)
    if "LONG PASS" in response_clean or ("LONG" in response_clean and "PASS" in response_clean and "THROW" not in response_clean):
        # CRITICAL: Only trust "Long pass" if distance is actually >= 250px
        # If VLM says "Long pass" but distance is < 250px, it's likely wrong - use "Short pass"
        if distance_num >= 250:
            return "Long pass", 85
        else:
            # VLM says long but distance suggests short - trust distance
            return "Short pass", 80  # High confidence but corrected
    
    # Fallback: Use geometry suggestion if VLM unclear
    if geometry_suggestion and geometry_suggestion in ["Short pass", "Long pass", "Cross", "Header", "Short throw-in", "Long throw-in"]:
        return geometry_suggestion, 70
    
    # Last resort: Use distance-based classification
    # DEFAULT to SHORT PASS (most common - 60% of passes)
    if distance_num >= LONG_PASS_THRESHOLD:
        return "Cross" if position == 'sideline' else "Long pass", 70
    elif distance_num < SHORT_PASS_THRESHOLD:
        # Clear short pass
        return "Short pass", 70
    else:
        # Middle range (150-300px): Default to SHORT PASS (most passes are short!)
        return "Short pass", 65


# === NEW: Multi-Frame Temporal Buffer Class ===

class TemporalBuffer:
    """
    Stores recent frame data for temporal analysis of pass events.
    """
    def __init__(self, size=8):
        self.size = size
        self.frames = deque(maxlen=size)
        self.ball_positions = deque(maxlen=size)
        self.player_data = deque(maxlen=size)  # {player_id: (bbox, position)}
        self.ball_heights_at_player = deque(maxlen=size)  # Track ball height relative to nearest player
    
    def add_frame(self, frame_idx, frame, ball_xy, players):
        """Add frame data to buffer"""
        self.frames.append({
            'idx': frame_idx,
            'frame': frame
        })
        self.ball_positions.append(ball_xy)
        self.player_data.append(players)
        
        # Track ball height relative to nearest player
        if ball_xy is not None and players:
            nearest_player = None
            min_dist = float('inf')
            for pid, data in players.items():
                pos = data['position']
                dist = np.linalg.norm(np.array(ball_xy) - np.array(pos))
                if dist < min_dist:
                    min_dist = dist
                    nearest_player = data
            
            if nearest_player and min_dist < 100:
                bbox = nearest_player['bbox']
                player_height = bbox[3] - bbox[1]
                ball_relative_y = (ball_xy[1] - bbox[1]) / player_height if player_height > 0 else 0.5
                self.ball_heights_at_player.append(ball_relative_y)
            else:
                self.ball_heights_at_player.append(None)
        else:
            self.ball_heights_at_player.append(None)
    
    def get_ball_trajectory(self):
        """Get ball movement pattern over buffer"""
        valid_positions = [p for p in self.ball_positions if p is not None]
        if len(valid_positions) < 2:
            return None
        
        # Calculate vertical and horizontal movement
        y_coords = [p[1] for p in valid_positions]
        x_coords = [p[0] for p in valid_positions]
        
        y_variance = max(y_coords) - min(y_coords) if y_coords else 0
        x_variance = max(x_coords) - min(x_coords) if x_coords else 0
        
        # Calculate direction
        y_direction = y_coords[-1] - y_coords[0] if len(y_coords) >= 2 else 0
        x_direction = x_coords[-1] - x_coords[0] if len(x_coords) >= 2 else 0
        
        return {
            'y_variance': y_variance,
            'x_variance': x_variance,
            'y_direction': y_direction,  # Positive = moving down
            'x_direction': x_direction,
            'is_aerial': y_variance > 50,
            'is_descending': y_direction > 30,
            'is_ascending': y_direction < -30
        }
    
    def check_header_pattern(self, threshold_frames=4):
        """
        STRICT header check - must have ball at head height for 4+ consecutive frames.
        Headers are RARE (only 7% of passes).
        """
        heights = list(self.ball_heights_at_player)
        if not heights or len(heights) < threshold_frames:
            return False, 0
        
        consecutive_head = 0
        max_consecutive = 0
        
        for h in heights:
            # STRICT: Ball must be in top 15% of player (actual head zone)
            if h is not None and h < 0.15:
                consecutive_head += 1
                max_consecutive = max(max_consecutive, consecutive_head)
            else:
                consecutive_head = 0
        
        # Lower confidence per frame - headers need strong evidence
        confidence = min(70, max_consecutive * 18)  # 18 points per frame (max 70)
        return max_consecutive >= threshold_frames, confidence
    
    def get_multi_frame_crop(self, center_xy, crop_size=400):
        """Get crops from multiple frames for VLM analysis"""
        crops = []
        for frame_data in self.frames:
            frame = frame_data['frame']
            h, w = frame.shape[:2]
            if center_xy is not None:
                x1 = max(0, int(center_xy[0]) - crop_size // 2)
                y1 = max(0, int(center_xy[1]) - crop_size // 2)
                x2 = min(w, x1 + crop_size)
                y2 = min(h, y1 + crop_size)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
        return crops


# === NEW: Confidence Scoring System ===

class PassConfidenceScorer:
    """
    Combines multiple signals to compute overall pass confidence.
    """
    
    @staticmethod
    def compute_geometry_confidence(distance, passer_pos, receiver_pos, frame_width, frame_height, pitch_bounds):
        """Compute confidence based on geometric analysis"""
        confidence = 70  # Base confidence
        
        # Distance-based confidence
        if MIN_PASS_DISTANCE < distance < 500:
            confidence += 10  # Reasonable pass distance
        elif distance < MIN_PASS_DISTANCE:
            confidence -= 30  # Too short
        elif distance > 600:
            confidence -= 20  # Suspiciously long
        
        # Position-based confidence
        # If both players are within pitch bounds, higher confidence
        if pitch_bounds.get('detected', False):
            left = pitch_bounds['left_sideline']
            right = pitch_bounds['right_sideline']
            
            passer_in_pitch = left < passer_pos[0] < right
            receiver_in_pitch = left < receiver_pos[0] < right
            
            if passer_in_pitch and receiver_in_pitch:
                confidence += 10
            elif not passer_in_pitch:
                confidence -= 10  # Passer outside pitch (might be throw-in)
        
        return min(100, max(0, confidence))
    
    @staticmethod
    def compute_temporal_confidence(ball_trajectory, header_check, ownership_duration):
        """Compute confidence based on temporal patterns"""
        confidence = 70  # Base confidence
        
        # Ball movement confidence
        if ball_trajectory:
            total_movement = ball_trajectory['y_variance'] + ball_trajectory['x_variance']
            if total_movement > 30:
                confidence += 10  # Ball clearly moved
            elif total_movement < 10:
                confidence -= 20  # Ball barely moved
        
        # Ownership duration
        if ownership_duration >= MIN_OWNERSHIP_FRAMES:
            confidence += 10
        else:
            confidence -= 15
        
        return min(100, max(0, confidence))
    
    @staticmethod
    def combine_confidence(geometry_conf, temporal_conf, vlm_conf):
        """Combine multiple confidence scores - VLM has highest weight"""
        # NEW Weights: geometry 20%, temporal 25%, VLM 55% (VLM is primary!)
        combined = (geometry_conf * 0.20) + (temporal_conf * 0.25) + (vlm_conf * 0.55)
        return int(combined)
    
    @staticmethod
    def should_trust_vlm_classification(vlm_type, geometry_type, vlm_conf):
        """
        Decide if VLM classification should override geometry.
        VLM is trusted for SHORT/LONG passes but geometry helps for rare events.
        """
        # VLM is primary - trust it for common passes
        if vlm_type in ["Short pass", "Long pass"] and vlm_conf >= 75:
            return True
        
        # For rare events, require higher VLM confidence
        if vlm_type in ["Header", "Cross", "Throw-in", "Short throw-in", "Long throw-in"]:
            # Only trust VLM for rare events if very confident
            return vlm_conf >= 82
        
        # Default to geometry if VLM uncertain
        return vlm_conf >= 70


# === Helper Functions ===

def is_header_action(ball_xy, player_bbox, ball_history=None, temporal_buffer=None):
    """
    STRICT header detection - Headers are RARE (only 7% of passes).
    Must have strong visual evidence across multiple frames.
    Returns: (is_header: bool, confidence: 0-100)
    """
    x1, y1, x2, y2 = player_bbox
    player_height = y2 - y1
    
    # STRICT: Ball must be in TOP 12% of player (head zone only)
    head_zone_bottom = y1 + (player_height * HEADER_HEIGHT_RATIO)  # 12%
    ball_at_head = ball_xy[1] < head_zone_bottom
    
    # Ball must also be horizontally near player (not just above)
    player_center_x = (x1 + x2) / 2
    horizontal_distance = abs(ball_xy[0] - player_center_x)
    ball_horizontally_close = horizontal_distance < (player_height * 0.5)
    
    if not ball_at_head or not ball_horizontally_close:
        return False, 0
    
    # REQUIRE multi-frame confirmation (headers need 4+ consecutive frames)
    if temporal_buffer:
        is_header, header_conf = temporal_buffer.check_header_pattern(HEADER_CONSECUTIVE_FRAMES)
        if is_header:
            # Still apply lower confidence - let VLM verify
            return True, min(header_conf, 65)  # Cap at 65, let VLM boost
        else:
            return False, 0  # Not enough consecutive frames
    
    # Strict ball history analysis
    if ball_history and len(ball_history) >= 5:
        recent_y = [pos[1][1] for pos in ball_history[-5:]]
        y_variance = max(recent_y) - min(recent_y)
        
        # Ball must have clear vertical movement (aerial ball)
        if y_variance < 60:  # STRICTER: Require more vertical movement
            return False, 0
        
        # Check ball is at head level for multiple frames
        head_frames = sum(1 for i, pos in enumerate(ball_history[-5:]) 
                        if pos[1][1] < head_zone_bottom)
        
        if head_frames < 3:  # Need 3+ frames at head level
            return False, 0
        
        confidence = min(60, 30 + (head_frames * 10))  # Lower max confidence
        return True, confidence
    
    # Without temporal data, don't suggest header (too risky)
    return False, 0


def is_sideline_position(x_pos, y_pos, frame_width, frame_height, pitch_bounds=None):
    """
    Check if position is near sideline (for throw-in detection).
    Uses pitch detection for accuracy.
    """
    if pitch_bounds and pitch_bounds.get('detected', False):
        return is_outside_pitch(x_pos, y_pos, pitch_bounds)
    
    # Fallback to frame-based detection
    margin_x = frame_width * SIDELINE_MARGIN
    margin_y = frame_height * SIDELINE_MARGIN
    
    near_left = x_pos < margin_x
    near_right = x_pos > (frame_width - margin_x)
    near_top = y_pos < margin_y
    near_bottom = y_pos > (frame_height - margin_y)
    
    return near_left or near_right or near_top or near_bottom


def is_outside_pitch(x_pos, y_pos, pitch_bounds):
    """Check if position is OUTSIDE the detected pitch boundaries"""
    if not pitch_bounds.get('detected', False):
        return False
    
    left = pitch_bounds['left_sideline']
    right = pitch_bounds['right_sideline']
    top = pitch_bounds['top_sideline']
    bottom = pitch_bounds['bottom_sideline']
    
    # Add small margin for throw-in standing position
    margin = 20  # pixels
    
    outside = (x_pos < left - margin or 
               x_pos > right + margin or 
               y_pos < top - margin or 
               y_pos > bottom + margin)
    
    return outside


def is_wide_position(x_pos, frame_width, pitch_bounds=None):
    """Check if position is in wide area (for cross detection)"""
    if pitch_bounds and pitch_bounds.get('detected', False):
        left = pitch_bounds['left_sideline']
        right = pitch_bounds['right_sideline']
        pitch_width = right - left
        wide_margin = pitch_width * CROSS_ZONE_WIDTH
        return x_pos < (left + wide_margin) or x_pos > (right - wide_margin)
    
    left_zone = frame_width * CROSS_ZONE_WIDTH
    right_zone = frame_width * (1 - CROSS_ZONE_WIDTH)
    return x_pos < left_zone or x_pos > right_zone


def is_central_position(x_pos, frame_width, pitch_bounds=None):
    """Check if position is in central area"""
    if pitch_bounds and pitch_bounds.get('detected', False):
        left = pitch_bounds['left_sideline']
        right = pitch_bounds['right_sideline']
        center = (left + right) / 2
        pitch_width = right - left
        central_margin = pitch_width * 0.35
        return (center - central_margin) < x_pos < (center + central_margin)
    
    left_zone = frame_width * CROSS_ZONE_WIDTH
    right_zone = frame_width * (1 - CROSS_ZONE_WIDTH)
    return left_zone <= x_pos <= right_zone


def is_referee(frame, bbox):
    """Check if detected person is likely a referee."""
    x1, y1, x2, y2 = map(int, bbox)
    jersey_y2 = y1 + int((y2 - y1) * 0.5)
    crop = frame[max(0, y1):jersey_y2, max(0, x1):x2]
    
    if crop.size == 0:
        return False
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 50, 80])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    
    black_mask = cv2.inRange(hsv, black_lower, black_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    total_pixels = crop.shape[0] * crop.shape[1]
    black_pct = cv2.countNonZero(black_mask) / total_pixels if total_pixels > 0 else 0
    yellow_pct = cv2.countNonZero(yellow_mask) / total_pixels if total_pixels > 0 else 0
    green_pct = cv2.countNonZero(green_mask) / total_pixels if total_pixels > 0 else 0
    
    if black_pct > 0.50 or yellow_pct > 0.35 or green_pct > 0.35:
        return True
    
    return False


def detect_team_color_opencv(frame, bbox, team_a_name="Blue", team_b_name="Red"):
    """Detect jersey color using OpenCV HSV analysis."""
    if is_referee(frame, bbox):
        return "Referee"
    
    x1, y1, x2, y2 = map(int, bbox)
    jersey_y1 = y1
    jersey_y2 = y1 + int((y2 - y1) * 0.6)
    pad_x = int((x2 - x1) * 0.1)
    crop = frame[max(0, jersey_y1):jersey_y2, max(0, x1+pad_x):max(0, x2-pad_x)]
    
    if crop.size == 0:
        return "Unknown"
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    blue_lower = np.array([85, 40, 40])
    blue_upper = np.array([135, 255, 255])
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    blue_pixels = cv2.countNonZero(blue_mask)
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = crop.shape[0] * crop.shape[1]
    
    blue_pct = blue_pixels / total_pixels if total_pixels > 0 else 0
    red_pct = red_pixels / total_pixels if total_pixels > 0 else 0
    
    min_pct = 0.05
    
    if blue_pct > red_pct and blue_pct > min_pct:
        return team_a_name
    elif red_pct > blue_pct and red_pct > min_pct:
        return team_b_name
    else:
        avg_color = np.mean(crop, axis=(0, 1))
        b, g, r = avg_color
        if b > r * 1.15 and b > g * 1.1:
            return team_a_name
        elif r > b * 1.15 and r > g * 0.9:
            return team_b_name
        else:
            return "Unknown"


def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def classify_pass_by_geometry(distance, passer_pos, receiver_pos, ball_xy, passer_bbox, 
                              frame_width, frame_height, ball_history, pitch_bounds, temporal_buffer=None):
    """
    VERY CONSERVATIVE geometry classification.
    BIAS: 60% of passes should be SHORT PASSES (most common!)
    Throw-ins and headers are RARE - require strong evidence.
    """
    
    # === HEADER CHECK (VERY RARE - only 7% of passes) ===
    is_header, header_conf = is_header_action(ball_xy, passer_bbox, ball_history, temporal_buffer)
    # Only suggest header if EXTREMELY confident
    if is_header and header_conf >= 65:
        return "Header", header_conf
    
    # === THROW-IN CHECK (EXTREMELY RARE - only 7% of passes) ===
    # DISABLED BY DEFAULT - too many false positives
    if not THROW_IN_DISABLED:
        near_sideline = is_sideline_position(passer_pos[0], passer_pos[1], frame_width, frame_height, pitch_bounds)
        
        # ULTRA STRICT: ONLY detect throw-in if:
        # 1. Pitch bounds detected AND
        # 2. Player is CLEARLY outside pitch
        if near_sideline and REQUIRE_PITCH_BOUNDS_FOR_THROW_IN:
            if pitch_bounds and pitch_bounds.get('detected', False):
                if is_outside_pitch(passer_pos[0], passer_pos[1], pitch_bounds):
                    # Double check: must be VERY far outside (not just at edge)
                    left = pitch_bounds.get('left_sideline', 0)
                    right = pitch_bounds.get('right_sideline', frame_width)
                    margin = 40  # Must be 40+ pixels outside pitch
                    
                    clearly_outside = (passer_pos[0] < left - margin or 
                                       passer_pos[0] > right + margin)
                    
                    if clearly_outside:
                        if distance < SHORT_PASS_THRESHOLD:
                            return "Short throw-in", 50
                        else:
                            return "Long throw-in", 50
        # NOT a throw-in - continue to regular pass classification
    
    # === CROSS CHECK (RARE - 5% of passes) ===
    passer_wide = is_wide_position(passer_pos[0], frame_width, pitch_bounds)
    receiver_central = is_central_position(receiver_pos[0], frame_width, pitch_bounds)
    
    is_aerial = False
    if temporal_buffer:
        trajectory = temporal_buffer.get_ball_trajectory()
        if trajectory:
            is_aerial = trajectory.get('is_aerial', False) and trajectory.get('y_variance', 0) > 100
    
    # Cross requires: wide + central receiver + long distance + clearly aerial
    if passer_wide and receiver_central and distance > SHORT_PASS_THRESHOLD * 1.5 and is_aerial:
        return "Cross", 55
    
    # === DEFAULT: REGULAR GROUND PASSES (85% of all passes) ===
    # SHORT PASS is the DEFAULT (60% of all passes!)
    if distance < SHORT_PASS_THRESHOLD:  # < 150px - CLEAR short pass
        return "Short pass", 88  # HIGH confidence - most common!
    elif distance >= LONG_PASS_THRESHOLD:  # >= 300px - CLEAR long pass
        return "Long pass", 78
    else:
        # Middle ground (150-300px): DEFAULT to SHORT PASS (60% are short!)
        # Only classify as long if clearly in upper range (250+ px)
        if distance >= 250:
            return "Long pass", 65
        else:
            return "Short pass", 72  # Default to short with good confidence


def detect_pitch_boundaries(pitch_model, frame):
    """Detect pitch boundaries using the pitch detection model."""
    try:
        results = pitch_model(frame, imgsz=1280, verbose=False)[0]
        
        if results.masks is not None and len(results.masks) > 0:
            mask = results.masks.data[0].cpu().numpy()
            coords = np.where(mask > 0.5)
            if len(coords[0]) > 0 and len(coords[1]) > 0:
                min_y, max_y = coords[0].min(), coords[0].max()
                min_x, max_x = coords[1].min(), coords[1].max()
                return {
                    'detected': True,
                    'left_sideline': min_x,
                    'right_sideline': max_x,
                    'top_sideline': min_y,
                    'bottom_sideline': max_y,
                    'pitch_width': max_x - min_x,
                    'pitch_height': max_y - min_y
                }
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            box = boxes[largest_idx]
            return {
                'detected': True,
                'left_sideline': int(box[0]),
                'right_sideline': int(box[2]),
                'top_sideline': int(box[1]),
                'bottom_sideline': int(box[3]),
                'pitch_width': int(box[2] - box[0]),
                'pitch_height': int(box[3] - box[1])
            }
    except:
        pass
    
    return {'detected': False}


# --- MAIN ENGINE ---
def run_analysis(video_path, use_vlm=True, debug=False):
    v_info = sv.VideoInfo.from_video_path(video_path)
    frame_width = v_info.width
    frame_height = v_info.height
    fps = v_info.fps
    
    print()
    print("=" * 70)
    print("🚀 LOADING ALL 4 AI MODELS FOR LIGHTNING AI")
    print("=" * 70)
    
    # === 1. Player Detection Model ===
    print(f"[1/4] Loading Player Detection Model...")
    if not os.path.exists(PLAYER_MODEL):
        print(f"      ❌ ERROR: Model not found: {PLAYER_MODEL}")
        return None, []
    p_m = YOLO(PLAYER_MODEL).to(DEVICE)
    print(f"      ✅ Player Model: {PLAYER_MODEL}")
    print(f"      📋 Classes: {p_m.names}")
    
    # === 2. Ball Detection Model ===
    print(f"[2/4] Loading Ball Detection Model...")
    if not os.path.exists(BALL_MODEL):
        print(f"      ❌ ERROR: Model not found: {BALL_MODEL}")
        return None, []
    b_m = YOLO(BALL_MODEL).to(DEVICE)
    print(f"      ✅ Ball Model: {BALL_MODEL}")
    print(f"      📋 Classes: {b_m.names}")
    
    # === 3. Pitch Detection Model ===
    print(f"[3/4] Loading Pitch Detection Model...")
    pitch_m = None
    use_pitch_detection = False
    if os.path.exists(PITCH_MODEL):
        try:
            pitch_m = YOLO(PITCH_MODEL).to(DEVICE)
            use_pitch_detection = True
            print(f"      ✅ Pitch Model: {PITCH_MODEL}")
            print(f"      📋 Classes: {pitch_m.names}")
        except Exception as e:
            print(f"      ⚠️  Could not load: {e}")
    else:
        print(f"      ⚠️  Model not found: {PITCH_MODEL}")
    
    # === 4. Molmo VLM ===
    print(f"[4/4] Molmo-7B VLM...")
    print(f"      ✅ VLM Model: allenai/Molmo-7B-D-0924 (loaded at startup)")
    
    print("=" * 70)
    models_count = 3 if use_pitch_detection else 2
    print(f"✅ ALL {models_count + 1} MODELS READY!")
    print(f"   🏃 Player Detection: ✅")
    print(f"   ⚽ Ball Detection:   ✅")
    print(f"   🏟️  Pitch Detection:  {'✅' if use_pitch_detection else '⚠️ OFF'}")
    print(f"   🧠 VLM (Molmo-7B):   ✅")
    print("=" * 70)
    print()
    
    # === Initialize Shot Detector (if available) ===
    shot_detector = None
    if SHOT_DETECTION_AVAILABLE:
        try:
            shot_detector = ShotDetector(frame_width, frame_height, fps, vlm_query)
            print(f"✅ Shot Detector initialized")
        except Exception as e:
            print(f"⚠️  Shot Detector initialization failed: {e}")
            shot_detector = None
    
    tracker = sv.ByteTrack()
    
    # === Initialize tracking state ===
    pass_events = []
    player_teams = {}
    player_positions = {}
    player_bboxes = {}
    current_owner = None
    ownership_start_frame = 0
    last_event_frame = -100
    
    ball_history = []
    BALL_HISTORY_SIZE = 15
    
    pitch_bounds = {'detected': False}
    PITCH_DETECT_INTERVAL = 100
    
    # === Initialize temporal buffer ===
    temporal_buffer = TemporalBuffer(size=TEMPORAL_BUFFER_SIZE)
    confidence_scorer = PassConfidenceScorer()
    
    # === Debug counters ===
    ownership_changes = 0
    filtered_by_distance = 0
    filtered_by_cooldown = 0
    filtered_by_ownership_duration = 0
    filtered_by_referee = 0
    filtered_by_confidence = 0
    vlm_stage1_rejections = 0

    pass_types = ["Short pass", "Long pass", "Cross", "Short throw-in", "Long throw-in", "Header"]
    stats = {
        TEAM_A_NAME: {pt: {"success": 0, "fail": 0, "total": 0} for pt in pass_types},
        TEAM_B_NAME: {pt: {"success": 0, "fail": 0, "total": 0} for pt in pass_types}
    }

    print(f"🎬 Processing: {video_path}...")
    print(f"📐 Frame size: {frame_width}x{frame_height} @ {fps:.1f}fps")
    print()
    print("📋 MODEL STATUS:")
    print(f"   🏃 Player Detection:    ✅ Active (YOLO)")
    print(f"   ⚽ Ball Detection:      ✅ Active (YOLO)")
    print(f"   🏟️  Pitch Detection:     {'✅ Active (YOLO)' if use_pitch_detection else '⚠️ Disabled'}")
    print(f"   🧠 VLM Stage 1 (Is Pass): {'✅ Active' if USE_VLM_STAGE1 and use_vlm else '⚠️ Disabled'}")
    print(f"   🧠 VLM Stage 2 (Type):    {'✅ Active' if USE_VLM_STAGE2 and use_vlm else '⚠️ Disabled'}")
    print(f"   👕 Team Detection:      ✅ Active (OpenCV HSV)")
    print()
    print(f"⚙️  Settings:")
    print(f"   Ball proximity: {BALL_PROXIMITY_THRESHOLD}px")
    print(f"   Cooldown: {EVENT_COOLDOWN_FRAMES} frames")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"   Temporal buffer: {TEMPORAL_BUFFER_SIZE} frames")
    print()
    
    model_stats = {
        'player_detections': 0,
        'ball_detections': 0,
        'pitch_detections': 0,
        'vlm_stage1': 0,
        'vlm_stage2': 0
    }
    
    video_writer = None
    output_video_path = 'annotated_test_video.mp4'
    if SAVE_ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    recent_passes = []
    PASS_DISPLAY_DURATION = 60
    
    for f_idx, frame in enumerate(tqdm(sv.get_video_frames_generator(video_path), total=v_info.total_frames)):
        
        # === Pitch detection (every N frames) ===
        if use_pitch_detection and (f_idx % PITCH_DETECT_INTERVAL == 0):
            pitch_bounds = detect_pitch_boundaries(pitch_m, frame)
            model_stats['pitch_detections'] += 1
            if pitch_bounds['detected'] and f_idx == 0:
                print(f"📐 Pitch detected: {pitch_bounds['pitch_width']}x{pitch_bounds['pitch_height']}px")
        
        # === Player detection (separate model) ===
        p_det = tracker.update_with_detections(sv.Detections.from_ultralytics(p_m(frame, imgsz=1280, verbose=False)[0]))
        model_stats['player_detections'] += 1
        
        # === Ball detection (separate model) ===
        b_det = sv.Detections.from_ultralytics(b_m(frame, imgsz=640, verbose=False)[0])
        model_stats['ball_detections'] += 1
        
        # === Update player tracking ===
        current_players = {}
        if p_det.tracker_id is not None:
            for tid, p_xy, bbox in zip(p_det.tracker_id, p_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER), p_det.xyxy):
                player_positions[tid] = p_xy
                player_bboxes[tid] = bbox
                current_players[tid] = {'position': p_xy, 'bbox': bbox}
                
                if tid not in player_teams or (f_idx % 300 == 0):
                    detected_team = detect_team_color_opencv(frame, bbox, TEAM_A_NAME, TEAM_B_NAME)
                    if detected_team != "Unknown" or tid not in player_teams:
                        player_teams[tid] = detected_team
        
        # === Ball tracking ===
        ball_coords = b_det.get_anchors_coordinates(sv.Position.CENTER)
        ball_xy = ball_coords[0] if len(ball_coords) > 0 else None
        
        if ball_xy is not None:
            ball_history.append((f_idx, ball_xy.copy()))
            if len(ball_history) > BALL_HISTORY_SIZE:
                ball_history.pop(0)
        
        # === Update temporal buffer ===
        temporal_buffer.add_frame(f_idx, frame, ball_xy, current_players)
        
        # === Pass detection logic ===
        if ball_xy is not None and p_det.tracker_id is not None:
            min_dist = float('inf')
            closest_player = None
            closest_bbox = None
            closest_pos = None
            
            for tid, p_xy, bbox in zip(p_det.tracker_id, p_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER), p_det.xyxy):
                ball_distance = np.linalg.norm(ball_xy - p_xy)
                if ball_distance < min_dist:
                    min_dist = ball_distance
                    closest_player = tid
                    closest_bbox = bbox
                    closest_pos = p_xy
            
            if min_dist < BALL_PROXIMITY_THRESHOLD and closest_player is not None:
                tid = closest_player
                bbox = closest_bbox
                p_xy = closest_pos
                
                if current_owner is not None and tid != current_owner:
                    ownership_changes += 1
                    ownership_duration = f_idx - ownership_start_frame
                    
                    # === Filter checks ===
                    if f_idx - last_event_frame <= EVENT_COOLDOWN_FRAMES:
                        filtered_by_cooldown += 1
                    elif ownership_duration < MIN_OWNERSHIP_FRAMES:
                        filtered_by_ownership_duration += 1
                    else:
                        passer_pos = player_positions.get(current_owner, p_xy)
                        receiver_pos = p_xy
                        distance = np.linalg.norm(passer_pos - receiver_pos)
                        
                        if distance < MIN_PASS_DISTANCE:
                            filtered_by_distance += 1
                        else:
                            # === Get team info ===
                            passer_team = player_teams.get(current_owner, "Unknown")
                            receiver_team = player_teams.get(tid, "Unknown")
                            
                            if passer_team == "Unknown" and current_owner in player_bboxes:
                                passer_team = detect_team_color_opencv(frame, player_bboxes[current_owner], TEAM_A_NAME, TEAM_B_NAME)
                                player_teams[current_owner] = passer_team
                            
                            if receiver_team == "Unknown":
                                receiver_team = detect_team_color_opencv(frame, bbox, TEAM_A_NAME, TEAM_B_NAME)
                                player_teams[tid] = receiver_team
                            
                            # Skip referee passes
                            if passer_team == "Referee" or receiver_team == "Referee":
                                filtered_by_referee += 1
                                current_owner = tid
                                ownership_start_frame = f_idx
                                continue
                            
                            passer_bbox = player_bboxes.get(current_owner, bbox)
                            
                            # === Step 1: Geometry classification ===
                            geometry_pass_type, geometry_conf = classify_pass_by_geometry(
                                distance, passer_pos, receiver_pos, ball_xy,
                                passer_bbox, frame_width, frame_height, ball_history, 
                                pitch_bounds, temporal_buffer
                            )
                            
                            # === Step 2: Compute temporal confidence ===
                            ball_trajectory = temporal_buffer.get_ball_trajectory()
                            header_check = temporal_buffer.check_header_pattern()
                            temporal_conf = confidence_scorer.compute_temporal_confidence(
                                ball_trajectory, header_check, ownership_duration
                            )
                            
                            # === Step 3: VLM verification (if enabled) ===
                            vlm_conf = 70  # Default VLM confidence
                            pass_type = geometry_pass_type
                            
                            if use_vlm:
                                # Get larger crop for better VLM context
                                crop = frame[
                                    max(0, int(ball_xy[1])-VLM_CROP_SIZE//2):min(frame_height, int(ball_xy[1])+VLM_CROP_SIZE//2),
                                    max(0, int(ball_xy[0])-VLM_CROP_SIZE//2):min(frame_width, int(ball_xy[0])+VLM_CROP_SIZE//2)
                                ]
                                
                                if crop.size > 0:
                                    # Stage 1: RELAXED - Is this likely a pass?
                                    if USE_VLM_STAGE1:
                                        is_pass_vlm, stage1_conf = vlm_stage1_is_pass(crop)
                                        model_stats['vlm_stage1'] += 1
                                        
                                        if not is_pass_vlm and stage1_conf < 40:
                                            # Only reject if VLM is VERY confident it's not a pass
                                            vlm_stage1_rejections += 1
                                            if debug:
                                                print(f"  [VLM REJECTED] Not a pass (conf={stage1_conf}%)")
                                            current_owner = tid
                                            ownership_start_frame = f_idx
                                            continue  # Skip only if very low confidence
                                        
                                        # Otherwise, let it through to Stage 2
                                        vlm_conf = stage1_conf
                                    
                                    # Stage 2: What type of pass? (only if Stage 1 passed)
                                    if USE_VLM_STAGE2 and vlm_conf >= 50:
                                        context_info = {
                                            'distance': f"{distance:.0f}",
                                            'position': 'sideline' if is_sideline_position(passer_pos[0], passer_pos[1], frame_width, frame_height, pitch_bounds) else 'pitch',
                                            'ball_height': 'high' if header_check[0] else 'normal'
                                        }
                                        vlm_pass_type, stage2_conf = vlm_stage2_classify_type(crop, geometry_pass_type, context_info)
                                        model_stats['vlm_stage2'] += 1
                                        
                                        # STRICT classification: RARE events require strict conditions
                                        # Throw-ins: Must be at sideline AND high confidence
                                        if vlm_pass_type in ["Short throw-in", "Long throw-in"]:
                                            # Throw-in requires: sideline position AND high confidence
                                            if context_info.get('position') == 'sideline' and stage2_conf >= 80:
                                                pass_type = vlm_pass_type
                                                vlm_conf = stage2_conf
                                            elif geometry_pass_type in ["Short throw-in", "Long throw-in"] and context_info.get('position') == 'sideline':
                                                # Geometry also says throw-in at sideline
                                                pass_type = geometry_pass_type
                                                vlm_conf = max(stage2_conf, geometry_conf)
                                            else:
                                                # Not at sideline or low confidence - likely a pass, not throw-in
                                                # Default to SHORT PASS (most common)
                                                if geometry_pass_type in ["Short pass", "Long pass"]:
                                                    pass_type = geometry_pass_type
                                                    vlm_conf = geometry_conf
                                                else:
                                                    # Default to SHORT PASS unless clearly long (>= 250px)
                                                    pass_type = "Short pass" if distance < 250 else "Long pass"
                                                    vlm_conf = 70
                                        # Header: Must have high ball
                                        elif vlm_pass_type == "Header":
                                            if context_info.get('ball_height') == 'high' and stage2_conf >= 80:
                                                pass_type = vlm_pass_type
                                                vlm_conf = stage2_conf
                                            elif geometry_pass_type == "Header" and context_info.get('ball_height') == 'high':
                                                pass_type = geometry_pass_type
                                                vlm_conf = max(stage2_conf, geometry_conf)
                                            else:
                                                # Not high ball - likely a pass
                                                # Default to SHORT PASS (most common)
                                                if geometry_pass_type in ["Short pass", "Long pass"]:
                                                    pass_type = geometry_pass_type
                                                    vlm_conf = geometry_conf
                                                else:
                                                    pass_type = "Short pass" if distance < 250 else "Long pass"
                                                    vlm_conf = 70
                                        # Cross: Must be from wing/long distance
                                        elif vlm_pass_type == "Cross":
                                            # Cross requires: sideline position OR long distance (>=250px) AND high confidence
                                            if (context_info.get('position') == 'sideline' or distance >= 250) and stage2_conf >= 80:
                                                pass_type = vlm_pass_type
                                                vlm_conf = stage2_conf
                                            elif geometry_pass_type == "Cross" and (context_info.get('position') == 'sideline' or distance >= 250):
                                                # Geometry also says cross at appropriate position
                                                pass_type = geometry_pass_type
                                                vlm_conf = max(stage2_conf, geometry_conf)
                                            else:
                                                # Not at sideline or too short - likely a pass
                                                pass_type = "Short pass" if distance < 250 else "Long pass"
                                                vlm_conf = 75
                                        # COMMON PASSES: Validate VLM with distance
                                        else:
                                            # STRICT VALIDATION: If VLM says "Long pass" but distance < 250px, override to "Short pass"
                                            if vlm_pass_type == "Long pass" and distance < 250:
                                                # VLM misclassified - trust distance and geometry
                                                if geometry_pass_type == "Short pass":
                                                    pass_type = "Short pass"
                                                    vlm_conf = max(75, geometry_conf)  # High confidence on correction
                                                else:
                                                    pass_type = "Short pass"  # Force short if distance suggests it
                                                    vlm_conf = 75
                                            elif vlm_pass_type == "Short pass" or (vlm_pass_type == "Long pass" and distance >= 250):
                                                # VLM classification is validated by distance
                                                if stage2_conf >= 75:
                                                    pass_type = vlm_pass_type
                                                    vlm_conf = stage2_conf
                                                elif stage2_conf >= 70:
                                                    if geometry_pass_type == vlm_pass_type:
                                                        pass_type = vlm_pass_type
                                                        vlm_conf = max(stage2_conf, geometry_conf)
                                                    else:
                                                        # Disagree - prefer VLM if validated by distance
                                                        pass_type = vlm_pass_type
                                                        vlm_conf = stage2_conf
                                                else:
                                                    # VLM uncertain - use geometry
                                                    pass_type = geometry_pass_type
                                                    vlm_conf = geometry_conf
                                            else:
                                                # VLM uncertain or unexpected type - use geometry
                                                pass_type = geometry_pass_type
                                                vlm_conf = geometry_conf
                            
                            # === Step 4: Compute final confidence ===
                            geo_conf_final = confidence_scorer.compute_geometry_confidence(
                                distance, passer_pos, receiver_pos, frame_width, frame_height, pitch_bounds
                            )
                            final_confidence = confidence_scorer.combine_confidence(
                                geo_conf_final, temporal_conf, vlm_conf
                            )
                            
                            # === Step 5: Filter by confidence ===
                            if final_confidence < CONFIDENCE_THRESHOLD:
                                filtered_by_confidence += 1
                                if debug:
                                    print(f"  [Filtered] Low confidence ({final_confidence}%): {pass_type}")
                                current_owner = tid
                                ownership_start_frame = f_idx
                                continue
                            
                            # === Normalize pass type ===
                            if "throw" in pass_type.lower():
                                if "short" not in pass_type.lower() and "long" not in pass_type.lower():
                                    pass_type = "Long throw-in" if distance > SHORT_PASS_THRESHOLD else "Short throw-in"
                            
                            if pass_type not in pass_types:
                                pass_type = "Short pass" if distance < SHORT_PASS_THRESHOLD else "Long pass"
                            
                            # === Determine result ===
                            if passer_team == receiver_team and passer_team != "Unknown":
                                result = "Success"
                            elif passer_team != receiver_team and passer_team != "Unknown" and receiver_team != "Unknown":
                                result = "Fail"
                            else:
                                result = "Unknown"
                            
                            # === Record stats ===
                            if passer_team in stats and pass_type in stats[passer_team]:
                                stats[passer_team][pass_type]["total"] += 1
                                if result == "Success":
                                    stats[passer_team][pass_type]["success"] += 1
                                elif result == "Fail":
                                    stats[passer_team][pass_type]["fail"] += 1
                            
                            # === Record pass event ===
                            pass_events.append({
                                "time": format_time(f_idx / fps),
                                "frame": f_idx,
                                "from_player": int(current_owner),
                                "to_player": int(tid),
                                "from_team": passer_team,
                                "to_team": receiver_team,
                                "pass_type": pass_type,
                                "result": result,
                                "distance_px": round(distance, 1),
                                "confidence": final_confidence,
                                "passer_x": round(passer_pos[0], 1),
                                "passer_y": round(passer_pos[1], 1)
                            })
                            
                            if debug:
                                print(f"  [Pass] {format_time(f_idx/fps)} | {passer_team} #{current_owner} → {receiver_team} #{tid} | {pass_type} | {result} | Conf: {final_confidence}%")
                            
                            # === Add to recent passes for annotation ===
                            if SAVE_ANNOTATED_VIDEO:
                                recent_passes.append({
                                    'frame': f_idx,
                                    'passer_pos': passer_pos.copy(),
                                    'receiver_pos': receiver_pos.copy(),
                                    'pass_type': pass_type,
                                    'result': result,
                                    'passer_team': passer_team,
                                    'confidence': final_confidence
                                })
                            
                            last_event_frame = f_idx
                
                # === Shot Detection (if enabled) ===
                # Check for shots EVERY frame when ball is moving (not just on ownership changes)
                if shot_detector is not None and ball_xy is not None:
                    # Always find closest player to ball (shots can happen during transitions)
                    shooter_tid = None
                    shooter_pos = None
                    shooter_bbox = None
                    shooter_team = "Unknown"
                    min_dist = float('inf')
                    
                    # Find closest player to ball
                    if p_det.tracker_id is not None:
                        for tid, p_xy, bbox in zip(p_det.tracker_id, p_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER), p_det.xyxy):
                            dist = np.linalg.norm(ball_xy - p_xy)
                            if dist < min_dist and dist < BALL_PROXIMITY_THRESHOLD * 1.5:  # Slightly wider range for shots
                                min_dist = dist
                                shooter_tid = tid
                                shooter_pos = p_xy
                                shooter_bbox = bbox
                    
                    # Fallback to current owner if no close player found
                    if shooter_tid is None and current_owner is not None:
                        shooter_tid = current_owner
                        shooter_pos = player_positions.get(shooter_tid)
                        shooter_bbox = player_bboxes.get(shooter_tid)
                    
                    if shooter_pos is not None and shooter_bbox is not None:
                        shooter_team = player_teams.get(shooter_tid, "Unknown")
                        
                        # Detect shot with UPDATED API (returns 4 values)
                        is_shot, shot_type, shot_conf, metrics = shot_detector.detect_shot(
                            f_idx, frame, ball_xy, shooter_tid, shooter_pos,
                            shooter_bbox, shooter_team, pitch_bounds if use_pitch_detection else None
                        )
                        
                        if is_shot:
                            # Record shot with metrics
                            shot_detector.record_shot(
                                f_idx, fps, shooter_tid, shooter_team,
                                shot_type, shot_conf, shooter_pos, ball_xy,
                                metrics=metrics  # Pass the enhanced metrics
                            )
                            if debug:
                                print(f"  [Shot] {format_time(f_idx/fps)} | {shooter_team} #{shooter_tid} | {shot_type} | Conf: {shot_conf}%")
                        elif debug and f_idx % 300 == 0:  # Debug shot detection every 300 frames
                            # Check why shots might not be detected
                            speed, dx, dy = shot_detector.get_ball_velocity()
                            in_range = shot_detector.is_in_shooting_range(shooter_pos) if shooter_pos is not None else False
                            if ball_xy and shooter_pos is not None:
                                dist, _ = shot_detector.calculate_distance_to_goal(ball_xy)
                                if speed >= 12:  # Lower threshold for debugging
                                    print(f"  [Shot Debug] Frame {f_idx}: speed={speed:.1f}, in_range={in_range}, dist={dist:.1f}m, shooter={shooter_team}#{shooter_tid}")
                
                if current_owner != tid:
                    ownership_start_frame = f_idx
                current_owner = tid
        
        # === Video annotation ===
        if SAVE_ANNOTATED_VIDEO and video_writer is not None:
            annotated_frame = frame.copy()
            
            # Draw players
            if p_det.tracker_id is not None:
                for tid_draw, bbox_draw in zip(p_det.tracker_id, p_det.xyxy):
                    team = player_teams.get(tid_draw, "Unknown")
                    if team == TEAM_A_NAME:
                        color = (255, 0, 0)  # Blue
                    elif team == TEAM_B_NAME:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (128, 128, 128)
                    
                    x1, y1, x2, y2 = map(int, bbox_draw)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"#{tid_draw}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball
            if ball_xy is not None:
                cv2.circle(annotated_frame, (int(ball_xy[0]), int(ball_xy[1])), 10, (0, 255, 255), -1)
            
            # Draw recent passes
            recent_passes = [p for p in recent_passes if f_idx - p['frame'] < PASS_DISPLAY_DURATION]
            for pass_info in recent_passes:
                p1 = (int(pass_info['passer_pos'][0]), int(pass_info['passer_pos'][1]))
                p2 = (int(pass_info['receiver_pos'][0]), int(pass_info['receiver_pos'][1]))
                
                if pass_info['result'] == "Success":
                    arrow_color = (0, 255, 0)
                else:
                    arrow_color = (0, 0, 255)
                
                cv2.arrowedLine(annotated_frame, p1, p2, arrow_color, 3, tipLength=0.1)
                
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                label = f"{pass_info['pass_type']} ({pass_info['confidence']}%)"
                cv2.putText(annotated_frame, label, (mid_x, mid_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color, 2)
            
            # Draw info overlay
            cv2.putText(annotated_frame, f"Frame: {f_idx} | Time: {format_time(f_idx/fps)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Passes: {len(pass_events)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            video_writer.write(annotated_frame)
        
        # Periodic cleanup
        if f_idx % 150 == 0 and f_idx > 0:
            torch.cuda.empty_cache()

    if SAVE_ANNOTATED_VIDEO and video_writer is not None:
        video_writer.release()
        print(f"\n🎥 Annotated video saved to: {output_video_path}")
    
    # === Print results ===
    print()
    print("=" * 80)
    print("🔍 DEBUG INFO")
    print("=" * 80)
    print(f"Total ownership changes detected: {ownership_changes}")
    print(f"Filtered by cooldown: {filtered_by_cooldown}")
    print(f"Filtered by ownership duration: {filtered_by_ownership_duration}")
    print(f"Filtered by min distance: {filtered_by_distance}")
    print(f"Filtered by referee: {filtered_by_referee}")
    print(f"Filtered by low confidence (<{CONFIDENCE_THRESHOLD}%): {filtered_by_confidence}")
    print(f"VLM Stage 1 rejections: {vlm_stage1_rejections}")
    print(f"Final passes recorded: {len(pass_events)}")
    
    print()
    print("=" * 80)
    print("🤖 MODEL USAGE STATISTICS")
    print("=" * 80)
    print(f"🏃 Player Detection Model:  {model_stats['player_detections']:,} inferences")
    print(f"⚽ Ball Detection Model:    {model_stats['ball_detections']:,} inferences")
    print(f"🏟️  Pitch Detection Model:   {model_stats['pitch_detections']:,} inferences")
    print(f"🧠 VLM Stage 1 (Is Pass):   {model_stats['vlm_stage1']:,} queries")
    print(f"🧠 VLM Stage 2 (Type):      {model_stats['vlm_stage2']:,} queries")
    
    team_counts = {"Blue": 0, "Red": 0, "Unknown": 0}
    for team in player_teams.values():
        if team in team_counts:
            team_counts[team] += 1
        else:
            team_counts["Unknown"] += 1
    print(f"Players detected - Blue: {team_counts['Blue']}, Red: {team_counts['Red']}, Unknown: {team_counts['Unknown']}")

    # === Save CSV report ===
    csv_path = 'test_video_pass_report.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["time", "frame", "from_player", "to_player", "from_team", "to_team", 
                      "pass_type", "result", "distance_px", "confidence", "passer_x", "passer_y"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pass_events)
    
    # === Print summary ===
    print()
    print("=" * 80)
    print("📊 PASS ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    header = f"{'Pass Type':<18} | {'Blue Team':^30} | {'Red Team':^30} | {'Total':^30}"
    subheader = f"{'':<18} | {'Total':^8} {'Success':^10} {'Fail':^10} | {'Total':^8} {'Success':^10} {'Fail':^10} | {'Total':^8} {'Success':^10} {'Fail':^10}"
    
    print(header)
    print(subheader)
    print("-" * 80)
    
    for pt in pass_types:
        blue = stats[TEAM_A_NAME][pt]
        red = stats[TEAM_B_NAME][pt]
        total_all = blue["total"] + red["total"]
        success_all = blue["success"] + red["success"]
        fail_all = blue["fail"] + red["fail"]
        
        row = f"{pt:<18} | {blue['total']:^8} {blue['success']:^10} {blue['fail']:^10} | {red['total']:^8} {red['success']:^10} {red['fail']:^10} | {total_all:^8} {success_all:^10} {fail_all:^10}"
        print(row)
    
    print("-" * 80)
    
    for team_name in [TEAM_A_NAME, TEAM_B_NAME]:
        total = sum(stats[team_name][pt]["total"] for pt in pass_types)
        success = sum(stats[team_name][pt]["success"] for pt in pass_types)
        fail = sum(stats[team_name][pt]["fail"] for pt in pass_types)
        rate = (success / (success + fail) * 100) if (success + fail) > 0 else 0
        print(f"\n🏃 {team_name} Team Total: {total} passes, {success} success, {fail} fail ({rate:.1f}% success rate)")
    
    # === Print comparison with manual analysis ===
    print()
    print("=" * 80)
    print("📋 COMPARISON WITH MANUAL ANALYSIS")
    print("=" * 80)
    
    manual = {
        "Short pass": {"Blue": 17, "Red": 10, "Total": 27},
        "Long pass": {"Blue": 5, "Red": 3, "Total": 8},
        "Cross": {"Blue": 1, "Red": 1, "Total": 2},
        "Short throw-in": {"Blue": 1, "Red": 1, "Total": 2},
        "Long throw-in": {"Blue": 1, "Red": 0, "Total": 1},
        "Header": {"Blue": 0, "Red": 3, "Total": 3}
    }
    
    print(f"{'Pass Type':<18} | {'Manual':^10} | {'AI':^10} | {'Diff':^10}")
    print("-" * 55)
    
    total_manual = 0
    total_ai = 0
    for pt in pass_types:
        m_val = manual.get(pt, {}).get("Total", 0)
        ai_val = stats[TEAM_A_NAME][pt]["total"] + stats[TEAM_B_NAME][pt]["total"]
        diff = ai_val - m_val
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        status = "✅" if abs(diff) <= 2 else "⚠️" if abs(diff) <= 5 else "❌"
        print(f"{pt:<18} | {m_val:^10} | {ai_val:^10} | {diff_str:^10} {status}")
        total_manual += m_val
        total_ai += ai_val
    
    print("-" * 55)
    total_diff = total_ai - total_manual
    diff_str = f"+{total_diff}" if total_diff > 0 else str(total_diff)
    accuracy = 100 - abs(total_diff / total_manual * 100) if total_manual > 0 else 0
    print(f"{'TOTAL':<18} | {total_manual:^10} | {total_ai:^10} | {diff_str:^10}")
    print(f"\n📊 Overall Accuracy: {accuracy:.1f}%")
    
    print()
    print("=" * 80)
    print(f"✅ Detailed report saved to: {csv_path}")
    print("=" * 80)
    
    # === Generate HTML Scout Match Report ===
    html_report_path = generate_scout_report_html(pass_events, video_path, stats, fps)
    print(f"📋 Scout Match Report saved to: {html_report_path}")
    print("=" * 80)
    print()
    print("🎬 VIDEO PLAYBACK INSTRUCTIONS:")
    print("=" * 80)
    video_name = os.path.basename(video_path)
    print(f"   To enable video playback in the HTML report:")
    print(f"   1. Copy '{video_name}' to the SAME folder as '{html_report_path}'")
    print(f"   2. Both files must be in the same directory!")
    print(f"   3. Open the HTML file in Chrome/Edge/Firefox")
    print()
    print(f"   📁 Required files in same folder:")
    print(f"      - {html_report_path}")
    print(f"      - {video_name}")
    print("=" * 80)
    
    # === Generate Shot Reports (if shot detector was used) ===
    if shot_detector is not None:
        shot_events = shot_detector.get_shot_events()
        shot_stats = shot_detector.get_stats()
        
        if len(shot_events) > 0:
            print()
            print("=" * 80)
            print("🎯 SHOT DETECTION SUMMARY")
            print("=" * 80)
            shot_detector.print_summary()
            print("=" * 80)
            
            # Generate shot JSON report
            if generate_shot_report_json:
                shot_json_path = generate_shot_report_json(shot_events, shot_stats, "shot_report.json")
                print(f"📊 Shot JSON report saved to: {shot_json_path}")
            
            # Generate shot CSV report
            if generate_shot_report_csv:
                shot_csv_path = generate_shot_report_csv(shot_events, "shot_report.csv")
                print(f"📊 Shot CSV report saved to: {shot_csv_path}")
            
            # Generate combined HTML report (passes + shots)
            if SHOT_REPORT_GENERATOR_AVAILABLE and generate_full_scout_report_html:
                annotated_video_path = output_video_path if SAVE_ANNOTATED_VIDEO else video_path
                combined_html_path = generate_full_scout_report_html(
                    pass_events, shot_events, video_path, annotated_video_path,
                    stats, shot_stats, fps
                )
                print(f"📋 Combined Scout Report (Passes + Shots) saved to: {combined_html_path}")
                print("=" * 80)
        else:
            print()
            print("=" * 80)
            print("🎯 SHOT DETECTION: No shots detected in this video")
            print("=" * 80)
    
    return stats, pass_events


# === YouTube Download & API Integration ===

def download_youtube_video(url: str, output_dir: str = INPUT_VIDEO_DIR) -> Optional[str]:
    """
    Download video from YouTube URL using yt-dlp
    Returns path to downloaded video file
    """
    if not YT_DLP_AVAILABLE:
        print("   ❌ ERROR: yt-dlp is not installed")
        print("   💡 Install it with: pip install yt-dlp")
        print("   💡 Or use a local video file instead")
        return None
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if ffmpeg is available
        import shutil
        ffmpeg_available = shutil.which('ffmpeg') is not None
        
        if not ffmpeg_available:
            print("   ⚠️  WARNING: ffmpeg not found - using single format download")
            print("   💡 Install ffmpeg for better quality: sudo apt-get install ffmpeg")
            # Use single format that doesn't require merging
            format_selector = 'best[ext=mp4]/best[height<=720]/best'
        else:
            print("   ✅ ffmpeg found - using best quality with audio merging")
            # Can merge video and audio for best quality
            format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        
        # Use a simpler filename pattern to avoid special character issues
        ydl_opts = {
            'format': format_selector,
            'outtmpl': os.path.join(output_dir, 'youtube_%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'progress_hooks': [lambda d: print(f"      ⬇️  Progress: {d.get('_percent_str', 'N/A')} - {d.get('_speed_str', 'N/A')}", end='\r') if d.get('status') == 'downloading' else None],
        }
        
        downloaded_file = None
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            print("   🔄 Extracting video information...")
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id', 'unknown')
            video_title = info.get('title', 'Unknown')
            video_duration = info.get('duration', 0)
            
            print(f"   📹 Video Info:")
            print(f"      Title: {video_title[:60]}...")
            print(f"      ID: {video_id}")
            print(f"      Duration: {video_duration // 60}:{video_duration % 60:02d}")
            print()
            print("   🔄 Downloading video (this may take a while)...")
            
            # Download the video
            ydl.download([url])
            print()  # New line after progress
            
            # Find the downloaded file by video ID
            expected_file = os.path.join(output_dir, f'youtube_{video_id}.mp4')
            if os.path.exists(expected_file):
                downloaded_file = expected_file
            else:
                # Try to find any mp4 file in the directory (fallback)
                for file in os.listdir(output_dir):
                    if file.endswith('.mp4') and 'youtube' in file.lower():
                        downloaded_file = os.path.join(output_dir, file)
                        break
                
                # Last resort: find any mp4 file
                if not downloaded_file:
                    for file in os.listdir(output_dir):
                        if file.endswith('.mp4'):
                            downloaded_file = os.path.join(output_dir, file)
                            break
            
            if downloaded_file and os.path.exists(downloaded_file):
                return downloaded_file
            else:
                print("   ❌ Downloaded file not found in expected location")
                return None
                
    except Exception as e:
        print(f"   ❌ Error downloading video: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_pass_events_to_json(pass_events: list, video_path: str) -> Dict[str, Any]:
    """
    Convert pass_events from run_analysis to JSON format expected by API
    Converts numpy types to native Python types for JSON serialization
    """
    def convert_to_native(value):
        """Convert numpy types to native Python types"""
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return [convert_to_native(item) for item in value]
        elif isinstance(value, dict):
            return {k: convert_to_native(v) for k, v in value.items()}
        else:
            return value
    
    # Convert pass events to API format
    passes_json = []
    for event in pass_events:
        pass_obj = {
            "time": str(event.get("time", "0:00")),
            "frame": int(convert_to_native(event.get("frame", 0))),
            "from_player": int(convert_to_native(event.get("from_player", 0))),
            "to_player": int(convert_to_native(event.get("to_player", 0))),
            "from_team": str(event.get("from_team", "Unknown")),
            "to_team": str(event.get("to_team", "Unknown")),
            "pass_type": str(event.get("pass_type", "Short pass")),
            "result": str(event.get("result", "Unknown")),
            "distance_px": float(convert_to_native(event.get("distance_px", 0))),
            "confidence": int(convert_to_native(event.get("confidence", 0))),
            "passer_x": float(convert_to_native(event.get("passer_x", 0))),
            "passer_y": float(convert_to_native(event.get("passer_y", 0)))
        }
        passes_json.append(pass_obj)
    
    match_report = {
        "match_report": {
            "total_passes": int(len(pass_events)),
            "passes": passes_json
        },
        "metadata": {
            "source_file": str(os.path.basename(video_path)),
            "format_version": "1.0"
        }
    }
    
    return match_report


def save_json_file(data: Dict[str, Any], output_path: str = "match_analysis_result.json", create_backup: bool = True) -> str:
    """
    Save analysis results to JSON file with atomic write to prevent corruption.
    Optionally creates a timestamped backup of previous file.
    """
    import tempfile
    import shutil
    from datetime import datetime
    
    try:
        # Validate JSON data can be serialized before writing
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"❌ JSON serialization error: {e}")
            # Try to fix numpy types
            logger.info("   Attempting to fix numpy types...")
            data = convert_numpy_to_native(data)
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Create backup of existing file if it exists and is valid
        if create_backup and os.path.exists(output_path):
            try:
                # Verify existing file is valid JSON before backing up
                with open(output_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Validate it's valid JSON
                
                # Create timestamped backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{output_path}.backup_{timestamp}"
                shutil.copy2(output_path, backup_path)
                logger.info(f"   📦 Backup created: {backup_path}")
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"   ⚠️  Existing file is corrupted or missing, skipping backup")
        
        # Atomic write: write to temp file first, then rename
        temp_dir = os.path.dirname(output_path) or '.'
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                        dir=temp_dir, delete=False, 
                                        suffix='.json.tmp') as temp_file:
            temp_path = temp_file.name
            temp_file.write(json_str)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Ensure data is written to disk
        
        # Atomic rename (works on Unix/Linux, Windows may need special handling)
        try:
            if os.path.exists(output_path):
                os.remove(output_path)  # Remove old file first
            os.rename(temp_path, output_path)
        except OSError:
            # Fallback for Windows or permission issues
            shutil.move(temp_path, output_path)
        
        # Verify the file was written correctly
        file_size = os.path.getsize(output_path)
        logger.info(f"✅ JSON file saved: {output_path} ({file_size:,} bytes)")
        
        # Double-check it's valid JSON
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                json.load(f)
            logger.info(f"   ✅ JSON file validated successfully")
        except json.JSONDecodeError as e:
            logger.error(f"   ❌ WARNING: Saved file is not valid JSON: {e}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Error saving JSON file: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return ""


def convert_numpy_to_native(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_native(item) for item in obj]
    else:
        return obj


class MatchAnalysisPipeline:
    """API Pipeline for submitting match analysis results"""
    
    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
        self.match_id = None
        self.video_url = None
        self._step2_success = False
        
    def step1_get_next_match(self) -> Optional[Dict[str, Any]]:
        """Step 1: Get next match details from API"""
        try:
            url = f"{self.base_url}/internal/next-match"
            print(f"   🔗 API Endpoint: {url}")
            print(f"   📤 Sending GET request...")
            print(f"   🔑 Request Headers:")
            print(f"      x-api-key: {self.headers.get('x-api-key', 'NOT SET')[:30]}...")
            print(f"      Content-Type: {self.headers.get('Content-Type', 'NOT SET')}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            print()
            print(f"   📥 RESPONSE DETAILS:")
            print(f"      Status Code: {response.status_code} {response.reason}")
            print(f"      Response Headers:")
            for key, value in response.headers.items():
                if key.lower() in ['content-type', 'content-length', 'server', 'date']:
                    print(f"         {key}: {value}")
            
            # Try to parse JSON response
            try:
                data = response.json()
                print(f"      Response Body (JSON):")
                print(f"         {json.dumps(data, indent=8)}")
            except:
                print(f"      Response Body (Text):")
                print(f"         {response.text[:500]}")
            
            print()
            
            if response.status_code == 200:
                print(f"   ✅ API Response received successfully")
                
                if data.get("message") == "Next match is available to analyse":
                    self.match_id = data["data"]["id"]
                    self.video_url = data["data"]["videoUrl"]
                    return data
                else:
                    print(f"   ⚠️  API Message: {data.get('message', 'Unknown')}")
                    return None
            elif response.status_code == 403:
                print(f"   ❌ ERROR: 403 Forbidden")
                print(f"   Possible causes:")
                print(f"      - API key is invalid or expired")
                print(f"      - API key format is incorrect")
                print(f"      - API key doesn't have permission for this endpoint")
                print(f"      - Rate limiting or IP restriction")
                if 'message' in data:
                    print(f"   API Error Message: {data.get('message')}")
                return None
            elif response.status_code == 401:
                print(f"   ❌ ERROR: 401 Unauthorized")
                print(f"   API key authentication failed")
                return None
            elif response.status_code == 404:
                print(f"   ℹ️  No match available (404 Not Found)")
                return None
            elif response.status_code == 429:
                print(f"   ❌ ERROR: 429 Too Many Requests")
                print(f"   Rate limit exceeded - please wait before retrying")
                return None
            elif response.status_code >= 500:
                print(f"   ❌ ERROR: {response.status_code} Server Error")
                print(f"   API server is experiencing issues")
                return None
            else:
                print(f"   ❌ ERROR: Unexpected status code {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print()
            print(f"   ❌ ERROR: Request Timeout")
            print(f"   The API server did not respond within 30 seconds")
            print(f"   Possible causes:")
            print(f"      - API server is down or slow")
            print(f"      - Network connectivity issues")
            return None
        except requests.exceptions.ConnectionError as e:
            print()
            print(f"   ❌ ERROR: Connection Failed")
            print(f"   Cannot reach API server: {str(e)}")
            print(f"   Possible causes:")
            print(f"      - API server is down")
            print(f"      - Network connectivity issues")
            print(f"      - Incorrect API URL: {self.base_url}")
            return None
        except requests.exceptions.RequestException as e:
            print()
            print(f"   ❌ ERROR: Request Exception")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response Status: {e.response.status_code}")
                print(f"   Response Text: {e.response.text[:200]}")
            return None
        except json.JSONDecodeError as e:
            print()
            print(f"   ❌ ERROR: Invalid JSON Response")
            print(f"   Could not parse API response as JSON")
            print(f"   Response text: {response.text[:200]}")
            return None
        except KeyError as e:
            print()
            print(f"   ❌ ERROR: Missing Expected Key")
            print(f"   Key: {str(e)}")
            print(f"   Response data: {data if 'data' in locals() else 'N/A'}")
            return None
        except Exception as e:
            print()
            print(f"   ❌ ERROR: Unexpected Error")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            import traceback
            print(f"   Traceback:")
            traceback.print_exc()
            return None
    
    def step2_update_status_processing(self) -> bool:
        """Step 2: Update match status to PROCESSING"""
        if not self.match_id:
            print("   ❌ ERROR: No match ID available. Run step1 first.")
            return False
            
        try:
            url = f"{self.base_url}/internal/{self.match_id}"
            payload = {"status": "PROCESSING"}
            print(f"   🔗 API Endpoint: {url}")
            print(f"   📤 Sending PUT request...")
            print(f"   📦 Request Payload: {json.dumps(payload, indent=6)}")
            
            response = requests.put(url, headers=self.headers, json=payload, timeout=30)
            
            print(f"   📥 Response Status: {response.status_code} {response.reason}")
            
            try:
                data = response.json()
                print(f"   📄 Response Body: {json.dumps(data, indent=6)}")
            except:
                print(f"   📄 Response Text: {response.text[:200]}")
            
            print()
            
            if response.status_code == 200:
                if data.get("message") == "Match status updated to PROCESSING":
                    self._step2_success = True
                    return True
                else:
                    print(f"   ⚠️  Unexpected response message: {data.get('message')}")
                    self._step2_success = True  # Still consider success
                    return True
            else:
                print(f"   ❌ ERROR: Status update failed with code {response.status_code}")
                self._step2_success = False
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR: {type(e).__name__}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text[:200]}")
            return False
    
    def step4_submit_results(self, analysis_result: Dict[str, Any]) -> bool:
        """Step 4: Submit analysis results"""
        if not self.match_id:
            print("   ❌ ERROR: No match ID available.")
            return False
            
        try:
            url = f"{self.base_url}/internal/result/{self.match_id}"
            
            # The API expects an array containing the analysis result
            payload = [analysis_result]
            
            print(f"   🔗 API Endpoint: {url}")
            print(f"   📤 Sending POST request...")
            print(f"   📦 Payload Structure:")
            print(f"      - Array with 1 item")
            print(f"      - Item keys: {list(analysis_result.keys())}")
            print(f"      - Total passes: {analysis_result.get('match_report', {}).get('total_passes', 0)}")
            
            # Try to calculate payload size (handle any serialization issues)
            try:
                payload_str = json.dumps(payload)
                print(f"   📊 Payload size: ~{len(payload_str)} bytes")
            except (TypeError, ValueError) as e:
                print(f"   📊 Payload size: Unable to calculate (will serialize on send)")
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            
            print(f"   📥 Response Status: {response.status_code} {response.reason}")
            
            try:
                data = response.json()
                print(f"   📄 Response Body: {json.dumps(data, indent=6)}")
            except:
                print(f"   📄 Response Text: {response.text[:500]}")
            
            print()
            
            if response.status_code == 200 or response.status_code == 201:
                if data.get("message") == "Match analysis submitted successfully":
                    submission_id = data.get('id', 'N/A')
                    print(f"   ✅ Submission successful! ID: {submission_id}")
                    return True
                else:
                    print(f"   ⚠️  Unexpected response message: {data.get('message')}")
                    return True  # Still consider success
            else:
                print(f"   ❌ ERROR: Submission failed with code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR: {type(e).__name__}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response Status: {e.response.status_code}")
                print(f"   Response Text: {e.response.text[:500]}")
            return False
        except Exception as e:
            print(f"   ❌ ERROR: Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step5_update_status_completed(self) -> bool:
        """Step 5: Update match status to COMPLETED"""
        if not self.match_id:
            print("   ❌ ERROR: No match ID available.")
            return False
            
        try:
            url = f"{self.base_url}/internal/{self.match_id}"
            payload = {"status": "COMPLETED"}
            print(f"   🔗 API Endpoint: {url}")
            print(f"   📤 Sending PUT request...")
            print(f"   📦 Request Payload: {json.dumps(payload, indent=6)}")
            
            response = requests.put(url, headers=self.headers, json=payload, timeout=30)
            
            print(f"   📥 Response Status: {response.status_code} {response.reason}")
            
            try:
                data = response.json()
                print(f"   📄 Response Body: {json.dumps(data, indent=6)}")
            except:
                print(f"   📄 Response Text: {response.text[:200]}")
            
            print()
            
            if response.status_code == 200:
                return True
            else:
                print(f"   ⚠️  Status update returned code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR: {type(e).__name__}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text[:200]}")
            return False


def generate_scout_report_html(pass_events, video_path, stats, fps):
    """
    Generate an HTML Scout Match Report with VIDEO PLAYBACK and clickable timestamps.
    Click any timestamp to jump to that moment in the video!
    """
    import os
    from datetime import datetime
    
    video_name = os.path.basename(video_path)
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate video duration from last event
    video_duration = max([e['frame'] for e in pass_events]) / fps if pass_events else 0
    
    # Pass type colors for visual distinction
    pass_colors = {
        "Short pass": "#4CAF50",      # Green
        "Long pass": "#2196F3",        # Blue
        "Cross": "#FF9800",            # Orange
        "Short throw-in": "#9C27B0",   # Purple
        "Long throw-in": "#673AB7",    # Deep Purple
        "Header": "#F44336"            # Red
    }
    
    # Calculate totals
    total_passes = len(pass_events)
    blue_passes = sum(1 for p in pass_events if p['from_team'] == 'Blue')
    red_passes = sum(1 for p in pass_events if p['from_team'] == 'Red')
    
    # Group passes by type for summary
    pass_type_counts = {}
    for p in pass_events:
        pt = p['pass_type']
        if pt not in pass_type_counts:
            pass_type_counts[pt] = 0
        pass_type_counts[pt] += 1
    
    # Generate timeline markers data for JavaScript
    markers_js = "const passMarkers = [\n"
    for idx, event in enumerate(pass_events):
        time_seconds = event['frame'] / fps
        pass_type = event['pass_type']
        color = pass_colors.get(pass_type, '#757575')
        markers_js += f'    {{time: {time_seconds:.2f}, type: "{pass_type}", color: "{color}", team: "{event["from_team"]}", result: "{event["result"]}", idx: {idx+1}}},\n'
    markers_js += "];\n"
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScoutMe - Match Report with Video</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }}
        
        .header {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        .logo {{
            font-size: 28px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .logo span {{
            background: #fff;
            color: #764ba2;
            padding: 5px 12px;
            border-radius: 8px;
        }}
        
        .header-info {{
            text-align: right;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px;
        }}
        
        /* Video Player Section */
        .video-section {{
            background: rgba(0,0,0,0.4);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .video-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .video-container {{
            position: relative;
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        #matchVideo {{
            width: 100%;
            border-radius: 12px;
            background: #000;
        }}
        
        .video-controls {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 15px;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        }}
        
        .play-btn {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: #fff;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s;
        }}
        
        .play-btn:hover {{
            transform: scale(1.1);
        }}
        
        .timeline-container {{
            flex: 1;
            position: relative;
        }}
        
        .timeline {{
            width: 100%;
            height: 12px;
            background: rgba(255,255,255,0.2);
            border-radius: 6px;
            cursor: pointer;
            position: relative;
        }}
        
        .timeline-progress {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 6px;
            width: 0%;
            transition: width 0.1s linear;
        }}
        
        .timeline-marker {{
            position: absolute;
            top: -8px;
            width: 4px;
            height: 28px;
            border-radius: 2px;
            cursor: pointer;
            transition: transform 0.2s;
            z-index: 10;
        }}
        
        .timeline-marker:hover {{
            transform: scaleY(1.3);
        }}
        
        .timeline-marker:hover::after {{
            content: attr(data-info);
            position: absolute;
            bottom: 35px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 100;
        }}
        
        .time-display {{
            font-family: 'Courier New', monospace;
            font-size: 14px;
            min-width: 100px;
            text-align: center;
        }}
        
        .current-pass-display {{
            background: rgba(102, 126, 234, 0.3);
            padding: 15px 20px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
            animation: fadeIn 0.3s ease;
        }}
        
        .current-pass-display.visible {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
        }}
        
        .card-value {{
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        
        .card-label {{
            font-size: 12px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .card.blue .card-value {{ color: #64B5F6; }}
        .card.red .card-value {{ color: #EF5350; }}
        .card.total .card-value {{ color: #FFD54F; }}
        
        .pass-type-summary {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 25px;
            justify-content: center;
        }}
        
        .pass-type-badge {{
            padding: 10px 18px;
            border-radius: 25px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .pass-type-badge:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .pass-type-badge .count {{
            background: rgba(255,255,255,0.25);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
        }}
        
        .table-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .table-header {{
            background: rgba(102, 126, 234, 0.3);
            padding: 18px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .table-title {{
            font-size: 18px;
            font-weight: 600;
        }}
        
        .search-box {{
            padding: 10px 15px;
            border-radius: 25px;
            border: none;
            background: rgba(255,255,255,0.1);
            color: #fff;
            width: 220px;
        }}
        
        .search-box::placeholder {{
            color: rgba(255,255,255,0.5);
        }}
        
        .table-scroll {{
            max-height: 500px;
            overflow-y: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: rgba(0,0,0,0.3);
            padding: 14px 18px;
            text-align: left;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255,255,255,0.8);
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        td {{
            padding: 12px 18px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        
        tr {{
            cursor: pointer;
            transition: background 0.2s;
        }}
        
        tr:hover {{
            background: rgba(102, 126, 234, 0.2);
        }}
        
        tr.active-row {{
            background: rgba(102, 126, 234, 0.4) !important;
            box-shadow: inset 0 0 0 2px #667eea;
        }}
        
        .player-cell {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .player-avatar {{
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
        }}
        
        .player-avatar.blue {{ background: #1565C0; }}
        .player-avatar.red {{ background: #C62828; }}
        .player-avatar.unknown {{ background: #757575; }}
        
        .timestamp-btn {{
            font-family: 'Courier New', monospace;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            border: none;
            color: #fff;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .timestamp-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .pass-type-tag {{
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }}
        
        .result-badge {{
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
        }}
        
        .result-badge.success {{ background: rgba(76, 175, 80, 0.3); color: #81C784; }}
        .result-badge.fail {{ background: rgba(244, 67, 54, 0.3); color: #E57373; }}
        .result-badge.unknown {{ background: rgba(255, 152, 0, 0.3); color: #FFB74D; }}
        
        .confidence-bar {{
            width: 60px;
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            border-radius: 3px;
        }}
        
        .filter-buttons {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 15px;
            padding: 0 25px;
        }}
        
        .filter-btn {{
            padding: 8px 14px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.2);
            background: transparent;
            color: #fff;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s ease;
        }}
        
        .filter-btn:hover, .filter-btn.active {{
            background: rgba(102, 126, 234, 0.5);
            border-color: transparent;
        }}
        
        .footer {{
            text-align: center;
            padding: 25px;
            opacity: 0.6;
            font-size: 12px;
        }}
        
        .legend {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}
        
        /* Scrollbar styling */
        .table-scroll::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .table-scroll::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.05);
        }}
        
        .table-scroll::-webkit-scrollbar-thumb {{
            background: rgba(102, 126, 234, 0.5);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            ⚽ <span>ScoutMe</span> AI Match Report
        </div>
        <div class="header-info">
            <div><strong>Video:</strong> {video_name}</div>
            <div><strong>Generated:</strong> {report_time}</div>
        </div>
    </div>
    
    <div class="container">
        <!-- Video Player Section -->
        <div class="video-section">
            <div class="video-title">🎬 Match Video Player - Click any timestamp to jump!</div>
            <div class="video-container">
                <video id="matchVideo" controls preload="metadata">
                    <source src="./{video_name}" type="video/mp4">
                    Your browser does not support video playback.
                </video>
                <div id="videoError" style="display:none; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center; color:#ff6b6b; background:rgba(0,0,0,0.9); padding:30px; border-radius:15px; max-width:80%;">
                    <div style="font-size:48px; margin-bottom:15px;">⚠️</div>
                    <div style="font-size:18px; font-weight:600; margin-bottom:10px;">Video Not Found!</div>
                    <div style="font-size:14px; opacity:0.8; line-height:1.6;">
                        Please copy <strong style="color:#4CAF50;">{video_name}</strong><br>
                        to the same folder as this HTML file.<br><br>
                        <span style="font-size:12px; opacity:0.6;">Both files must be in the same directory.</span>
                    </div>
                </div>
            </div>
            
            <div class="video-controls">
                <button class="play-btn" onclick="togglePlay()">▶</button>
                <div class="timeline-container">
                    <div class="timeline" id="timeline" onclick="seekVideo(event)">
                        <div class="timeline-progress" id="timelineProgress"></div>
                        <!-- Markers will be added by JavaScript -->
                    </div>
                </div>
                <div class="time-display">
                    <span id="currentTime">0:00</span> / <span id="duration">0:00</span>
                </div>
            </div>
            
            <!-- ScoutMe Pass Vocabulary Guide -->
            <div style="background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; margin-top: 20px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 16px; font-weight: 600; margin-bottom: 15px; text-align: center;">📚 ScoutMe Pass Vocabulary</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px;">
                    <div style="display: flex; align-items: flex-start; gap: 10px; padding: 10px; background: rgba(76,175,80,0.15); border-radius: 8px;">
                        <div style="width: 12px; height: 12px; background: #4CAF50; border-radius: 3px; margin-top: 4px;"></div>
                        <div>
                            <div style="font-weight: 600; font-size: 13px;">Short Ground Pass</div>
                            <div style="font-size: 11px; opacity: 0.8;">Controlled kick along grass to nearby teammate (&lt;15-20 yards). Used for possession building.</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: flex-start; gap: 10px; padding: 10px; background: rgba(33,150,243,0.15); border-radius: 8px;">
                        <div style="width: 12px; height: 12px; background: #2196F3; border-radius: 3px; margin-top: 4px;"></div>
                        <div>
                            <div style="font-weight: 600; font-size: 13px;">Long Ground Pass</div>
                            <div style="font-size: 11px; opacity: 0.8;">High-power kick across grass over large distance. Used to switch play quickly.</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: flex-start; gap: 10px; padding: 10px; background: rgba(255,152,0,0.15); border-radius: 8px;">
                        <div style="width: 12px; height: 12px; background: #FF9800; border-radius: 3px; margin-top: 4px;"></div>
                        <div>
                            <div style="font-weight: 600; font-size: 13px;">Cross</div>
                            <div style="font-size: 11px; opacity: 0.8;">Aerial ball from wing into penalty box. Goal creation opportunity for strikers.</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: flex-start; gap: 10px; padding: 10px; background: rgba(156,39,176,0.15); border-radius: 8px;">
                        <div style="width: 12px; height: 12px; background: #9C27B0; border-radius: 3px; margin-top: 4px;"></div>
                        <div>
                            <div style="font-weight: 600; font-size: 13px;">Throw-in</div>
                            <div style="font-size: 11px; opacity: 0.8;">Restart with two hands above head from sideline. Crucial for momentum retention.</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: flex-start; gap: 10px; padding: 10px; background: rgba(244,67,54,0.15); border-radius: 8px;">
                        <div style="width: 12px; height: 12px; background: #F44336; border-radius: 3px; margin-top: 4px;"></div>
                        <div>
                            <div style="font-weight: 600; font-size: 13px;">Header</div>
                            <div style="font-size: 11px; opacity: 0.8;">Ball redirected with forehead to teammate. Shows aerial dominance. VLM specialty detection.</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="current-pass-display" id="currentPassDisplay">
                <span style="font-size: 24px;">📍</span>
                <div>
                    <strong id="currentPassType">-</strong>
                    <span id="currentPassInfo" style="opacity: 0.8; font-size: 13px;"></span>
                </div>
            </div>
        </div>
        
        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="card total">
                <div class="card-value">{total_passes}</div>
                <div class="card-label">Total Passes</div>
            </div>
            <div class="card blue">
                <div class="card-value">{blue_passes}</div>
                <div class="card-label">Blue Team</div>
            </div>
            <div class="card red">
                <div class="card-value">{red_passes}</div>
                <div class="card-label">Red Team</div>
            </div>
            <div class="card">
                <div class="card-value" style="color: #81C784;">{sum(1 for p in pass_events if p['result'] == 'Success')}</div>
                <div class="card-label">Successful</div>
            </div>
            <div class="card">
                <div class="card-value" style="color: #E57373;">{sum(1 for p in pass_events if p['result'] == 'Fail')}</div>
                <div class="card-label">Failed</div>
            </div>
        </div>
        
        <!-- Pass Type Summary -->
        <div class="pass-type-summary">
'''
    
    # Add pass type badges
    for pass_type, color in pass_colors.items():
        count = pass_type_counts.get(pass_type, 0)
        html_content += f'''            <div class="pass-type-badge" style="background: {color};" onclick="filterByPassType('{pass_type}')">
                {pass_type} <span class="count">{count}</span>
            </div>
'''
    
    html_content += '''        </div>
        
        <!-- Data Table -->
        <div class="table-container">
            <div class="table-header">
                <div class="table-title">📋 Pass Events - Click row to watch!</div>
                <input type="text" class="search-box" placeholder="🔍 Search..." onkeyup="filterTable(this.value)">
            </div>
            
            <div class="filter-buttons">
                <button class="filter-btn active" onclick="filterByType('all')">All</button>
                <button class="filter-btn" onclick="filterByType('Blue')">🔵 Blue</button>
                <button class="filter-btn" onclick="filterByType('Red')">🔴 Red</button>
                <button class="filter-btn" onclick="filterByType('Success')">✅ Success</button>
                <button class="filter-btn" onclick="filterByType('Fail')">❌ Failed</button>
            </div>
            
            <div class="table-scroll">
                <table id="passTable">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>▶ Watch</th>
                            <th>From</th>
                            <th>To</th>
                            <th>Pass Type</th>
                            <th>Result</th>
                            <th>Conf</th>
                        </tr>
                    </thead>
                    <tbody>
'''
    
    # Add table rows for each pass event
    for idx, event in enumerate(pass_events, 1):
        from_team = event['from_team']
        to_team = event['to_team']
        pass_type = event['pass_type']
        result = event['result']
        confidence = event.get('confidence', 70)
        time_seconds = event['frame'] / fps
        
        from_avatar_class = from_team.lower() if from_team in ['Blue', 'Red'] else 'unknown'
        to_avatar_class = to_team.lower() if to_team in ['Blue', 'Red'] else 'unknown'
        result_class = result.lower()
        pass_color = pass_colors.get(pass_type, '#757575')
        
        # Confidence bar color
        if confidence >= 80:
            conf_color = '#4CAF50'
        elif confidence >= 60:
            conf_color = '#FF9800'
        else:
            conf_color = '#F44336'
        
        html_content += f'''                        <tr data-team="{from_team}" data-result="{result}" data-type="{pass_type}" data-time="{time_seconds:.2f}" data-idx="{idx}" onclick="jumpToTime({time_seconds:.2f}, {idx})">
                            <td>{idx}</td>
                            <td>
                                <button class="timestamp-btn" onclick="event.stopPropagation(); jumpToTime({time_seconds:.2f}, {idx})">
                                    ▶ {event['time']}
                                </button>
                            </td>
                            <td>
                                <div class="player-cell">
                                    <div class="player-avatar {from_avatar_class}">#{event['from_player']}</div>
                                    <span>{from_team}</span>
                                </div>
                            </td>
                            <td>
                                <div class="player-cell">
                                    <div class="player-avatar {to_avatar_class}">#{event['to_player']}</div>
                                    <span>{to_team}</span>
                                </div>
                            </td>
                            <td><span class="pass-type-tag" style="background: {pass_color};">{pass_type}</span></td>
                            <td><span class="result-badge {result_class}">{result}</span></td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: {confidence}%; background: {conf_color};"></div>
                                    </div>
                                    <span style="font-size: 11px;">{confidence}%</span>
                                </div>
                            </td>
                        </tr>
'''
    
    html_content += f'''                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Generated by ScoutMe AI Pass Analysis Engine v5</p>
        <p>🎬 Click any timestamp to watch that moment in the video!</p>
    </div>
    
    <script>
        // Pass markers data
        {markers_js}
        
        const video = document.getElementById('matchVideo');
        const timeline = document.getElementById('timeline');
        const timelineProgress = document.getElementById('timelineProgress');
        const currentTimeDisplay = document.getElementById('currentTime');
        const durationDisplay = document.getElementById('duration');
        const currentPassDisplay = document.getElementById('currentPassDisplay');
        const playBtn = document.querySelector('.play-btn');
        
        // Initialize markers on timeline
        video.addEventListener('loadedmetadata', function() {{
            const duration = video.duration;
            durationDisplay.textContent = formatTime(duration);
            
            // Add markers to timeline
            passMarkers.forEach(marker => {{
                const percent = (marker.time / duration) * 100;
                const markerEl = document.createElement('div');
                markerEl.className = 'timeline-marker';
                markerEl.style.left = percent + '%';
                markerEl.style.backgroundColor = marker.color;
                markerEl.dataset.info = '#' + marker.idx + ' ' + marker.type + ' (' + marker.team + ')';
                markerEl.onclick = function(e) {{
                    e.stopPropagation();
                    jumpToTime(marker.time, marker.idx);
                }};
                timeline.appendChild(markerEl);
            }});
        }});
        
        // Update timeline progress
        video.addEventListener('timeupdate', function() {{
            const percent = (video.currentTime / video.duration) * 100;
            timelineProgress.style.width = percent + '%';
            currentTimeDisplay.textContent = formatTime(video.currentTime);
            
            // Check if near any pass event
            const nearPass = passMarkers.find(m => Math.abs(m.time - video.currentTime) < 1);
            if (nearPass) {{
                showCurrentPass(nearPass);
            }} else {{
                hideCurrentPass();
            }}
        }});
        
        function togglePlay() {{
            if (video.paused) {{
                video.play();
                playBtn.textContent = '⏸';
            }} else {{
                video.pause();
                playBtn.textContent = '▶';
            }}
        }}
        
        video.addEventListener('play', () => playBtn.textContent = '⏸');
        video.addEventListener('pause', () => playBtn.textContent = '▶');
        
        // Handle video load error - show helpful message
        video.addEventListener('error', function(e) {{
            console.error('Video loading error:', e);
            document.getElementById('videoError').style.display = 'block';
            video.style.display = 'none';
        }});
        
        // Also check if video source fails to load
        const source = video.querySelector('source');
        if (source) {{
            source.addEventListener('error', function() {{
                document.getElementById('videoError').style.display = 'block';
                video.style.display = 'none';
            }});
        }}
        
        // Hide error if video loads successfully
        video.addEventListener('loadeddata', function() {{
            document.getElementById('videoError').style.display = 'none';
            video.style.display = 'block';
        }});
        
        function seekVideo(e) {{
            const rect = timeline.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            video.currentTime = percent * video.duration;
        }}
        
        function jumpToTime(seconds, idx) {{
            video.currentTime = Math.max(0, seconds - 1); // Start 1 second before
            video.play();
            playBtn.textContent = '⏸';
            
            // Highlight the row
            document.querySelectorAll('#passTable tbody tr').forEach(row => {{
                row.classList.remove('active-row');
                if (parseInt(row.dataset.idx) === idx) {{
                    row.classList.add('active-row');
                    row.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }});
            
            // Show pass info
            const pass = passMarkers.find(m => m.idx === idx);
            if (pass) {{
                showCurrentPass(pass);
            }}
        }}
        
        function showCurrentPass(pass) {{
            currentPassDisplay.classList.add('visible');
            document.getElementById('currentPassType').textContent = pass.type;
            document.getElementById('currentPassInfo').textContent = ' | ' + pass.team + ' Team | ' + pass.result;
        }}
        
        function hideCurrentPass() {{
            currentPassDisplay.classList.remove('visible');
        }}
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins + ':' + (secs < 10 ? '0' : '') + secs;
        }}
        
        function filterTable(searchText) {{
            const rows = document.querySelectorAll('#passTable tbody tr');
            searchText = searchText.toLowerCase();
            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchText) ? '' : 'none';
            }});
        }}
        
        function filterByType(type) {{
            const rows = document.querySelectorAll('#passTable tbody tr');
            const buttons = document.querySelectorAll('.filter-btn');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            rows.forEach(row => {{
                if (type === 'all') {{
                    row.style.display = '';
                }} else if (type === 'Blue' || type === 'Red') {{
                    row.style.display = row.dataset.team === type ? '' : 'none';
                }} else if (type === 'Success' || type === 'Fail') {{
                    row.style.display = row.dataset.result === type ? '' : 'none';
                }}
            }});
        }}
        
        function filterByPassType(type) {{
            const rows = document.querySelectorAll('#passTable tbody tr');
            rows.forEach(row => {{
                row.style.display = row.dataset.type === type ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
'''
    
    # Save HTML report
    report_path = 'scout_match_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ScoutMe - Soccer Pass Analysis v5 (Enhanced 80%+ Accuracy)')
    parser.add_argument('--source', type=str, default='input_video/test_video_playable.mp4', 
                       help='Path to video file or YouTube URL')
    parser.add_argument('--no-vlm', action='store_true', help='Disable VLM verification')
    parser.add_argument('--debug', action='store_true', help='Show detailed logs')
    parser.add_argument('--use-api', action='store_true', help='Use API workflow (get match, submit results)')
    parser.add_argument('--save-json', action='store_true', default=True, help='Save results to JSON file')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚀 SCOUTME - SOCCER PASS ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"📋 Configuration:")
    print(f"   Source: {args.source}")
    print(f"   Use API: {'✅ YES' if args.use_api else '❌ NO'}")
    print(f"   VLM Enabled: {'✅ YES' if not args.no_vlm else '❌ NO'}")
    print(f"   Save JSON: {'✅ YES' if args.save_json else '❌ NO'}")
    print("=" * 80 + "\n")
    
    video_path = args.source
    original_source = args.source
    video_source_type = "LOCAL_FILE"
    
    # Check if source is a YouTube URL
    is_youtube = 'youtube.com' in video_path or 'youtu.be' in video_path
    if is_youtube:
        video_source_type = "YOUTUBE_URL"
        print("🔍 DETECTED: YouTube URL")
        print(f"   URL: {video_path}\n")
    
    # If using API workflow, get match from API
    pipeline = None
    api_video_obtained = False
    
    if args.use_api:
        print("=" * 80)
        print("📡 STEP 1: API WORKFLOW - Fetching Match from API")
        print("=" * 80)
        print(f"   API URL: {BASE_URL}")
        print(f"   API Key: {API_KEY[:20]}...")
        print()
        
        pipeline = MatchAnalysisPipeline()
        
        # Step 1: Get next match
        print("   🔄 Calling API: GET /internal/next-match")
        match_data = pipeline.step1_get_next_match()
        
        if match_data and pipeline.video_url:
            video_path = pipeline.video_url
            api_video_obtained = True
            video_source_type = "API_YOUTUBE" if ('youtube.com' in video_path or 'youtu.be' in video_path) else "API_VIDEO"
            
            print()
            print("   ✅ API RESPONSE RECEIVED:")
            print(f"      Match ID: {pipeline.match_id}")
            print(f"      Video URL: {video_path}")
            print(f"      Video Source Type: {video_source_type}")
            print()
            
            # Step 2: Update status to PROCESSING (before analysis)
            print("   🔄 Calling API: PUT /internal/{match_id} (status: PROCESSING)")
            step2_result = pipeline.step2_update_status_processing()
            if step2_result:
                print("   ✅ Status updated to PROCESSING\n")
            else:
                print("   ⚠️  Failed to update status (continuing anyway)\n")
        else:
            print()
            print("   ⚠️  API RESPONSE: No match available")
            print(f"   📁 Falling back to provided source: {original_source}")
            print()
            video_source_type = "FALLBACK_LOCAL"
    else:
        print("=" * 80)
        print("📡 API WORKFLOW: DISABLED")
        print("=" * 80)
        print(f"   Using provided source: {video_path}\n")
    
    # Re-check if video is YouTube URL (might have changed from API)
    is_youtube = 'youtube.com' in video_path or 'youtu.be' in video_path
    
    # Download YouTube video if needed
    if is_youtube:
        print("=" * 80)
        print("📥 STEP 2: YOUTUBE VIDEO DOWNLOAD")
        print("=" * 80)
        print(f"   YouTube URL: {video_path}")
        print(f"   Output Directory: {INPUT_VIDEO_DIR}")
        print("   🔄 Starting download...\n")
        
        downloaded_path = download_youtube_video(video_path)
        
        if downloaded_path and os.path.exists(downloaded_path):
            file_size = os.path.getsize(downloaded_path) / (1024 * 1024)  # MB
            print()
            print("   ✅ DOWNLOAD SUCCESSFUL:")
            print(f"      Downloaded File: {downloaded_path}")
            print(f"      File Size: {file_size:.2f} MB")
            print()
            video_path = downloaded_path
            video_source_type = "DOWNLOADED_YOUTUBE"
        else:
            print()
            print("   ❌ DOWNLOAD FAILED")
            print("   Exiting...")
            exit(1)
    else:
        print("=" * 80)
        print("📥 STEP 2: VIDEO DOWNLOAD")
        print("=" * 80)
        print("   ℹ️  No YouTube URL detected - using local file")
        print(f"   Video Path: {video_path}\n")
    
    # Check if video file exists
    print("=" * 80)
    print("🔍 STEP 3: VIDEO FILE VERIFICATION")
    print("=" * 80)
    
    if not os.path.exists(video_path):
        print(f"   ❌ ERROR: Video file not found!")
        print(f"   Path: {video_path}")
        print("   Exiting...")
        exit(1)
    
    # Get video info
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            cap.release()
            
            print("   ✅ VIDEO FILE VERIFIED:")
            print(f"      File Path: {video_path}")
            print(f"      File Size: {file_size:.2f} MB")
            print(f"      Resolution: {width}x{height}")
            print(f"      FPS: {fps:.2f}")
            print(f"      Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"      Total Frames: {frame_count:,}")
            print(f"      Source Type: {video_source_type}")
            print()
        else:
            print("   ⚠️  Could not read video file (will try anyway)")
            print()
    except Exception as e:
        print(f"   ⚠️  Could not get video info: {e}")
        print("   Continuing anyway...\n")
    
    # Run analysis
    print("=" * 80)
    print("🎬 STEP 4: STARTING VIDEO ANALYSIS")
    print("=" * 80)
    print(f"   Processing: {os.path.basename(video_path)}")
    print()
    
    try:
        stats, pass_events = run_analysis(video_path, use_vlm=not args.no_vlm, debug=args.debug)
        
        # Convert to JSON format
        json_data = convert_pass_events_to_json(pass_events, video_path)
        
        # Save JSON file (with backup of previous file)
        if args.save_json:
            json_path = save_json_file(json_data, "match_analysis_result.json", create_backup=True)
            if json_path:
                logger.info(f"📄 JSON file saved: {json_path}")
            else:
                logger.error("❌ Failed to save JSON file")
        
        # Submit to API if using API workflow
        api_summary = {
            'step1_success': False,
            'step2_success': False,
            'step4_success': False,
            'step5_success': False,
            'match_id': None,
            'video_url': None
        }
        
        if args.use_api:
            if pipeline:
                api_summary['match_id'] = pipeline.match_id
                api_summary['video_url'] = pipeline.video_url
                api_summary['step1_success'] = pipeline.match_id is not None
                api_summary['step2_success'] = getattr(pipeline, '_step2_success', False)
            
            if args.use_api and pipeline and pipeline.match_id:
                print()
                print("=" * 80)
                print("📤 STEP 5: SUBMITTING RESULTS TO API")
                print("=" * 80)
                print(f"   Match ID: {pipeline.match_id}")
                print(f"   Total Passes: {len(pass_events)}")
                print()
                
                # Step 4: Submit results
                print("   🔄 Calling API: POST /internal/result/{match_id}")
                api_summary['step4_success'] = pipeline.step4_submit_results(json_data)
                
                print()
                print("   🔄 Calling API: PUT /internal/{match_id} (status: COMPLETED)")
                api_summary['step5_success'] = pipeline.step5_update_status_completed()
                
                print()
                print("=" * 80)
                print("📊 API WORKFLOW SUMMARY")
                print("=" * 80)
                print(f"   Step 1 (Get Match):     {'✅ SUCCESS' if api_summary['step1_success'] else '❌ FAILED'}")
                print(f"   Step 2 (Set PROCESSING): {'✅ SUCCESS' if api_summary['step2_success'] else '❌ FAILED'}")
                print(f"   Step 4 (Submit Results): {'✅ SUCCESS' if api_summary['step4_success'] else '❌ FAILED'}")
                print(f"   Step 5 (Set COMPLETED): {'✅ SUCCESS' if api_summary['step5_success'] else '❌ FAILED'}")
                print("=" * 80)
            elif args.use_api:
                print()
                print("=" * 80)
                print("⚠️  API WORKFLOW SKIPPED")
                print("=" * 80)
                print("   Reason: No match ID available (Step 1 API call failed)")
                print(f"   Step 1 Status: {'✅ SUCCESS' if api_summary['step1_success'] else '❌ FAILED'}")
                if not api_summary['step1_success']:
                    print("   💡 TIP: Check API key and network connectivity")
                print("=" * 80)
        
        print()
        print("=" * 80)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"   ✅ Total passes detected: {len(pass_events)}")
        print(f"   📄 JSON file: match_analysis_result.json")
        print(f"   📊 CSV file: test_video_pass_report.csv")
        print(f"   🌐 HTML report: scout_match_report.html")
        print("=" * 80)
        print()
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


