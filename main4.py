"""
ScoutMe - Soccer Match Analysis Engine v6.0
===============================================
Features:
- Pass Detection & Classification (VLM + 3 YOLO models)
- Shot on Goal Detection (On Target / Off Target / Goal)
- Interactive HTML Report with VIDEO POPUP
- Popup plays ANNOTATED VIDEO with AI highlights

Usage:
    python main6.py --source input_video/video.mp4 --debug

Author: ScoutMe AI
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from collections import deque

# Check dependencies
try:
    import supervision as sv
    from ultralytics import YOLO
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install opencv-python ultralytics supervision transformers pillow torch torchvision")
    sys.exit(1)

# Import our modules
from shot_detection import ShotDetector, generate_shot_report_csv, generate_shot_report_json
from scout_report_generator import generate_full_scout_report_html

# === CONFIGURATION ===
DATA_DIR = 'data'
PLAYER_MODEL = os.path.join(DATA_DIR, 'football-player-detection.pt')
BALL_MODEL = os.path.join(DATA_DIR, 'football-ball-detection.pt')
PITCH_MODEL = os.path.join(DATA_DIR, 'football-pitch-detection.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Team names
TEAM_A_NAME = "Blue"
TEAM_B_NAME = "Red"

# === DETECTION PARAMETERS ===
BALL_PROXIMITY_THRESHOLD = 70
EVENT_COOLDOWN_FRAMES = 35
MIN_PASS_DISTANCE = 40
MIN_OWNERSHIP_FRAMES = 6
SIDELINE_MARGIN = 0.05
CROSS_ZONE_WIDTH = 0.25
HEADER_HEIGHT_RATIO = 0.25
SHORT_PASS_THRESHOLD = 250
LONG_PASS_THRESHOLD = 400
CONFIDENCE_THRESHOLD = 72
TEMPORAL_BUFFER_SIZE = 10

# VLM settings
USE_VLM = True
VLM_CROP_SIZE = 450

# Video annotation
SAVE_ANNOTATED_VIDEO = True

# Pass type colors for annotation
PASS_COLORS = {
    "Short pass": (76, 175, 80),      # Green
    "Long pass": (33, 150, 243),       # Blue
    "Cross": (255, 152, 0),            # Orange
    "Short throw-in": (156, 39, 176),  # Purple
    "Long throw-in": (103, 58, 183),   # Deep Purple
    "Header": (244, 67, 54)            # Red
}

SHOT_COLORS = {
    "Shot on target": (34, 197, 94),   # Green
    "Shot off target": (239, 68, 68),  # Red
    "Shot blocked": (245, 158, 11),    # Orange
    "Goal": (255, 215, 0)              # Gold
}


# === LOAD VLM MODEL ===
print("=" * 70)
print("🚀 SCOUTME v6 - LOADING AI MODELS")
print("=" * 70)

print("\n🧠 Loading Molmo-7B VLM...")
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
print("✅ VLM Model loaded!")


def vlm_query(frame_crop, prompt, max_tokens=50):
    """Generic VLM query function"""
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


def vlm_stage1_is_pass(frame_crop):
    """Stage 1: Is this a pass?"""
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
    
    is_pass = "YES" in response_upper or ("NO" not in response_upper and "BALL" in response_upper)
    confidence = 75 if is_pass else 55
    
    return is_pass, confidence


def vlm_stage2_classify_type(frame_crop, geometry_suggestion, context_info):
    """Stage 2: Classify pass type - FIXED VERSION"""
    distance = context_info.get('distance', 'unknown')
    position = context_info.get('position', 'unknown')
    
    # Simpler, more direct prompt that forces a single letter answer
    prompt = f"""Soccer pass classification. Distance: {distance}px. Position: {position}.

Look at this image and classify the pass type.

RULES:
- 60% of passes are SHORT PASS (foot kick, close distance)
- 20% of passes are LONG PASS (powerful foot kick, far distance)
- 5% are CROSS (wing to penalty area)
- 5% are THROW-IN (player uses BOTH HANDS from sideline)
- 5% are HEADER (ball touches player's HEAD)

Answer with ONLY ONE LETTER:
A = SHORT PASS
B = LONG PASS
C = CROSS
D = THROW-IN
E = HEADER

My answer:"""

    response = vlm_query(frame_crop, prompt, max_tokens=5)
    response_upper = response.upper().strip()
    
    # Extract ONLY the first letter A-E
    first_char = ''
    for c in response_upper:
        if c in 'ABCDE':
            first_char = c
            break
    
    # Debug: print VLM response
    # print(f"    [VLM Raw] '{response}' -> Letter: '{first_char}'")
    
    # STRICT letter-only parsing (no keyword matching!)
    if first_char == 'A':
        return "Short pass", 85
    elif first_char == 'B':
        return "Long pass", 82
    elif first_char == 'C':
        return "Cross", 80
    elif first_char == 'D':
        return "Throw-in", 82
    elif first_char == 'E':
        return "Header", 75  # Lower confidence for Header - require extra validation
    
    # If VLM doesn't give clear answer, TRUST GEOMETRY!
    return geometry_suggestion, 75


def detect_team_color(frame, bbox):
    """Detect team color using HSV analysis"""
    x1, y1, x2, y2 = map(int, bbox)
    jersey_y2 = y1 + int((y2 - y1) * 0.6)
    pad_x = int((x2 - x1) * 0.1)
    crop = frame[max(0, y1):jersey_y2, max(0, x1+pad_x):max(0, x2-pad_x)]
    
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
    
    if blue_pct > red_pct and blue_pct > 0.05:
        return TEAM_A_NAME
    elif red_pct > blue_pct and red_pct > 0.05:
        return TEAM_B_NAME
    else:
        return "Unknown"


def is_referee(frame, bbox):
    """Check if detected person is a referee"""
    x1, y1, x2, y2 = map(int, bbox)
    jersey_y2 = y1 + int((y2 - y1) * 0.5)
    crop = frame[max(0, y1):jersey_y2, max(0, x1):x2]
    
    if crop.size == 0:
        return False
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 80]))
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    
    total_pixels = crop.shape[0] * crop.shape[1]
    black_pct = cv2.countNonZero(black_mask) / total_pixels if total_pixels > 0 else 0
    yellow_pct = cv2.countNonZero(yellow_mask) / total_pixels if total_pixels > 0 else 0
    
    return black_pct > 0.50 or yellow_pct > 0.35


def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


class PlayerReID:
    """
    Re-identification system to maintain consistent player IDs (1-22).
    Saves player images with bounding boxes throughout the match.
    """
    def __init__(self, max_players=22):
        self.max_players = max_players
        self.track_to_permanent = {}  # ByteTrack ID -> Permanent ID
        self.permanent_to_track = {}  # Permanent ID -> ByteTrack ID
        self.player_features = {}     # Permanent ID -> visual features
        self.player_teams = {}        # Permanent ID -> team color
        self.next_permanent_id = 1
        self.player_images = {}       # Permanent ID -> best crop image
        self.player_appearances = {}  # Permanent ID -> frame count
    
    def get_permanent_id(self, track_id, team, bbox, frame=None):
        """Get or assign permanent ID for a tracked player."""
        
        # Already mapped?
        if track_id in self.track_to_permanent:
            perm_id = self.track_to_permanent[track_id]
            self.player_appearances[perm_id] = self.player_appearances.get(perm_id, 0) + 1
            
            # Update best image if frame provided
            if frame is not None and bbox is not None:
                self._update_player_image(perm_id, frame, bbox)
            
            return perm_id
        
        # New track - assign new permanent ID
        if self.next_permanent_id <= self.max_players:
            perm_id = self.next_permanent_id
            self.next_permanent_id += 1
        else:
            # Recycle least seen player ID
            perm_id = min(self.player_appearances, key=self.player_appearances.get, default=1)
        
        self.track_to_permanent[track_id] = perm_id
        self.permanent_to_track[perm_id] = track_id
        self.player_teams[perm_id] = team
        self.player_appearances[perm_id] = 1
        
        # Save player image
        if frame is not None and bbox is not None:
            self._update_player_image(perm_id, frame, bbox)
        
        return perm_id
    
    def _update_player_image(self, perm_id, frame, bbox):
        """Save/update player crop image with bounding box."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # Add padding
            pad = 10
            x1_crop = max(0, x1 - pad)
            y1_crop = max(0, y1 - pad)
            x2_crop = min(frame.shape[1], x2 + pad)
            y2_crop = min(frame.shape[0], y2 + pad)
            
            crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            
            if crop.size > 0:
                # Draw bounding box on crop
                crop_with_box = crop.copy()
                box_x1 = pad
                box_y1 = pad
                box_x2 = crop.shape[1] - pad
                box_y2 = crop.shape[0] - pad
                
                # Get team color for box
                team = self.player_teams.get(perm_id, "Unknown")
                box_color = (255, 0, 0) if team == TEAM_A_NAME else ((0, 0, 255) if team == TEAM_B_NAME else (128, 128, 128))
                
                cv2.rectangle(crop_with_box, (box_x1, box_y1), (box_x2, box_y2), box_color, 3)
                
                # Add player ID text
                cv2.putText(crop_with_box, f"#{perm_id}", (box_x1 + 5, box_y1 + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                cv2.putText(crop_with_box, team, (box_x1 + 5, box_y1 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                # Keep the best quality image (largest)
                if perm_id not in self.player_images or crop.size > self.player_images[perm_id][0].size:
                    self.player_images[perm_id] = (crop, crop_with_box)
        except Exception as e:
            pass
    
    def save_player_images(self, output_dir="player_images"):
        """Save all player images as JPG files with bounding boxes."""
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        for perm_id in sorted(self.player_images.keys()):
            _, crop_with_box = self.player_images[perm_id]
            team = self.player_teams.get(perm_id, "Unknown")
            filename = f"{output_dir}/player_{perm_id:02d}_{team}.jpg"
            
            try:
                cv2.imwrite(filename, crop_with_box)
                saved_count += 1
            except:
                pass
        
        return saved_count


def classify_pass_by_geometry(distance, passer_pos, receiver_pos, frame_width, frame_height):
    """Classify pass type by geometry"""
    
    # Sideline check for throw-in
    margin = frame_width * SIDELINE_MARGIN
    near_sideline = passer_pos[0] < margin or passer_pos[0] > (frame_width - margin)
    
    if near_sideline:
        if distance < SHORT_PASS_THRESHOLD:
            return "Short throw-in", 70
        else:
            return "Long throw-in", 70
    
    # Cross check (from wing)
    in_wing = passer_pos[0] < frame_width * CROSS_ZONE_WIDTH or passer_pos[0] > frame_width * (1 - CROSS_ZONE_WIDTH)
    receiver_central = frame_width * 0.3 < receiver_pos[0] < frame_width * 0.7
    
    if in_wing and receiver_central and distance > SHORT_PASS_THRESHOLD:
        return "Cross", 72
    
    # Long pass
    if distance > LONG_PASS_THRESHOLD:
        return "Long pass", 80
    
    # Short pass (default)
    if distance <= SHORT_PASS_THRESHOLD:
        return "Short pass", 85
    
    # Medium distance
    return "Long pass", 70


def annotate_frame(frame, pass_event=None, shot_event=None, players=None, ball_xy=None):
    """Annotate frame with pass/shot information"""
    annotated = frame.copy()
    
    # Draw ball if detected
    if ball_xy is not None:
        cv2.circle(annotated, (int(ball_xy[0]), int(ball_xy[1])), 12, (255, 255, 255), -1)
        cv2.circle(annotated, (int(ball_xy[0]), int(ball_xy[1])), 14, (0, 0, 0), 2)
    
    # Draw pass annotation
    if pass_event:
        event_type = pass_event.get('pass_type', 'Pass')
        color = PASS_COLORS.get(event_type, (255, 255, 255))
        color_bgr = (color[2], color[1], color[0])  # Convert RGB to BGR
        
        # Draw text box
        text = f"{event_type} | {pass_event.get('from_team', '?')} #{pass_event.get('from_player', '?')} -> #{pass_event.get('to_player', '?')}"
        cv2.rectangle(annotated, (20, 20), (600, 80), color_bgr, -1)
        cv2.putText(annotated, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw confidence
        conf = pass_event.get('confidence', 0)
        cv2.putText(annotated, f"Conf: {conf}%", (620, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
    
    # Draw shot annotation
    if shot_event:
        shot_type = shot_event.get('shot_type', 'Shot')
        color = SHOT_COLORS.get(shot_type, (255, 255, 255))
        color_bgr = (color[2], color[1], color[0])
        
        # Draw text box at bottom
        text = f"🎯 {shot_type} | {shot_event.get('team', '?')} #{shot_event.get('shooter_id', '?')}"
        h = frame.shape[0]
        cv2.rectangle(annotated, (20, h-80), (500, h-20), color_bgr, -1)
        cv2.putText(annotated, text, (30, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return annotated


def run_analysis(video_path, debug=False):
    """Main analysis function"""
    
    # Check video exists
    if not os.path.exists(video_path):
        raise Exception(f"Could not open video at {video_path}")
    
    # Load YOLO models
    print("\n" + "=" * 70)
    print("🚀 LOADING DETECTION MODELS")
    print("=" * 70)
    
    print("[1/3] Loading Player Detection Model...")
    p_m = YOLO(PLAYER_MODEL)
    print(f"✅ Player Model: {PLAYER_MODEL}")
    
    print("[2/3] Loading Ball Detection Model...")
    b_m = YOLO(BALL_MODEL)
    print(f"✅ Ball Model: {BALL_MODEL}")
    
    print("[3/3] Loading Pitch Detection Model...")
    pitch_m = YOLO(PITCH_MODEL) if os.path.exists(PITCH_MODEL) else None
    if pitch_m:
        print(f"✅ Pitch Model: {PITCH_MODEL}")
    else:
        print("⚠️ Pitch model not found, using frame-based detection")
    
    print("\n✅ ALL MODELS READY!")
    print("=" * 70)
    
    # Open video
    v_info = sv.VideoInfo.from_video_path(video_path)
    frame_width = v_info.width
    frame_height = v_info.height
    fps = v_info.fps
    total_frames = v_info.total_frames
    
    print(f"\n🎬 Processing: {video_path}")
    print(f"📐 Frame size: {frame_width}x{frame_height} @ {fps:.1f}fps")
    print(f"⏱️ Duration: {format_time(total_frames / fps)}")
    
    # Initialize tracker
    tracker = sv.ByteTrack()
    
    # Initialize shot detector
    shot_detector = ShotDetector(frame_width, frame_height, fps, vlm_query)
    
    # Initialize player re-identification (for consistent IDs 1-22)
    player_reid = PlayerReID(max_players=22)
    
    # State variables
    player_positions = {}
    player_bboxes = {}
    player_teams = {}
    ball_history = deque(maxlen=20)
    
    current_owner = None
    ownership_start_frame = 0
    last_pass_frame = -EVENT_COOLDOWN_FRAMES
    
    pass_events = []
    
    # Stats
    pass_types = ["Short pass", "Long pass", "Cross", "Short throw-in", "Long throw-in", "Header"]
    pass_stats = {
        TEAM_A_NAME: {pt: {"success": 0, "fail": 0, "total": 0} for pt in pass_types},
        TEAM_B_NAME: {pt: {"success": 0, "fail": 0, "total": 0} for pt in pass_types}
    }
    
    # Video writer for annotated output
    video_writer = None
    output_video_path = f"annotated_{os.path.basename(video_path).replace(' ', '_')}"
    if SAVE_ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Track recent events for annotation
    recent_pass = None
    recent_shot = None
    ANNOTATION_DURATION = 60  # frames
    
    # Process frames
    print("\n" + "=" * 70)
    print("🔍 ANALYZING VIDEO...")
    print("=" * 70 + "\n")
    
    for f_idx, frame in enumerate(tqdm(sv.get_video_frames_generator(video_path), total=total_frames)):
        
        # Clear old annotations
        if recent_pass and f_idx - recent_pass['frame'] > ANNOTATION_DURATION:
            recent_pass = None
        if recent_shot and f_idx - recent_shot['frame'] > ANNOTATION_DURATION:
            recent_shot = None
        
        # === DETECTION ===
        
        # Player detection
        p_det = tracker.update_with_detections(
            sv.Detections.from_ultralytics(p_m(frame, imgsz=1280, verbose=False)[0])
        )
        
        # Ball detection
        b_det = sv.Detections.from_ultralytics(b_m(frame, imgsz=640, verbose=False)[0])
        ball_coords = b_det.get_anchors_coordinates(sv.Position.CENTER)
        ball_xy = ball_coords[0] if len(ball_coords) > 0 else None
        
        if ball_xy is not None:
            ball_history.append((f_idx, ball_xy.copy()))
        
        # Update player tracking
        if p_det.tracker_id is not None:
            for tid, p_xy, bbox in zip(p_det.tracker_id, 
                                        p_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER),
                                        p_det.xyxy):
                player_positions[tid] = p_xy
                player_bboxes[tid] = bbox
                
                # Detect team color periodically
                if tid not in player_teams or f_idx % 300 == 0:
                    if not is_referee(frame, bbox):
                        player_teams[tid] = detect_team_color(frame, bbox)
                    else:
                        player_teams[tid] = "Referee"
                
                # Get permanent player ID (1-22) and save image
                team = player_teams.get(tid, "Unknown")
                if team not in ["Unknown", "Referee"]:
                    perm_id = player_reid.get_permanent_id(tid, team, bbox, frame)
        
        # === PASS DETECTION ===
        if ball_xy is not None and p_det.tracker_id is not None:
            # Find closest player to ball
            min_dist = float('inf')
            closest_player = None
            closest_pos = None
            
            for tid, p_xy in zip(p_det.tracker_id, p_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)):
                dist = np.linalg.norm(ball_xy - p_xy)
                if dist < min_dist:
                    min_dist = dist
                    closest_player = tid
                    closest_pos = p_xy
            
            if min_dist < BALL_PROXIMITY_THRESHOLD and closest_player is not None:
                tid = closest_player
                
                # Check for ownership change (potential pass)
                if current_owner is not None and tid != current_owner:
                    ownership_duration = f_idx - ownership_start_frame
                    
                    # Apply filters
                    if f_idx - last_pass_frame > EVENT_COOLDOWN_FRAMES and ownership_duration >= MIN_OWNERSHIP_FRAMES:
                        passer_pos = player_positions.get(current_owner, closest_pos)
                        receiver_pos = closest_pos
                        distance = np.linalg.norm(passer_pos - receiver_pos)
                        
                        if distance >= MIN_PASS_DISTANCE:
                            passer_team = player_teams.get(current_owner, "Unknown")
                            receiver_team = player_teams.get(tid, "Unknown")
                            
                            # Skip referee passes
                            if passer_team != "Referee" and receiver_team != "Referee":
                                
                                # Geometry classification
                                geometry_type, geometry_conf = classify_pass_by_geometry(
                                    distance, passer_pos, receiver_pos, frame_width, frame_height
                                )
                                
                                # VLM classification
                                pass_type = geometry_type
                                vlm_conf = geometry_conf
                                
                                if USE_VLM:
                                    crop = frame[
                                        max(0, int(ball_xy[1])-VLM_CROP_SIZE//2):min(frame_height, int(ball_xy[1])+VLM_CROP_SIZE//2),
                                        max(0, int(ball_xy[0])-VLM_CROP_SIZE//2):min(frame_width, int(ball_xy[0])+VLM_CROP_SIZE//2)
                                    ]
                                    
                                    if crop.size > 0:
                                        is_pass, stage1_conf = vlm_stage1_is_pass(crop)
                                        
                                        if is_pass or stage1_conf >= 40:
                                            context = {
                                                'distance': f"{distance:.0f}",
                                                'position': 'sideline' if passer_pos[0] < frame_width * SIDELINE_MARGIN or passer_pos[0] > frame_width * (1 - SIDELINE_MARGIN) else 'pitch',
                                                'ball_height': 'normal'
                                            }
                                            vlm_type, stage2_conf = vlm_stage2_classify_type(crop, geometry_type, context)
                                            
                                            # SMART DECISION LOGIC:
                                            # 1. For SHORT/LONG pass - trust VLM if confident
                                            # 2. For RARE events - require BOTH VLM and geometry to agree, OR very high VLM confidence
                                            
                                            if vlm_type in ["Short pass", "Long pass"]:
                                                # VLM agrees it's a ground pass - use it
                                                pass_type = vlm_type
                                                vlm_conf = stage2_conf
                                            
                                            elif vlm_type == "Header":
                                                # STRICT Header check: requires VLM confidence > 85%
                                                # AND distance must be short (headers are short-range)
                                                if stage2_conf >= 85 and distance < 300:
                                                    pass_type = vlm_type
                                                    vlm_conf = stage2_conf
                                                else:
                                                    # Not confident enough, use geometry
                                                    pass_type = geometry_type
                                                    vlm_conf = geometry_conf
                                            
                                            elif vlm_type in ["Cross", "Throw-in"]:
                                                # Cross/Throw-in: require VLM confidence > 80%
                                                if stage2_conf >= 80:
                                                    pass_type = vlm_type
                                                    vlm_conf = stage2_conf
                                                else:
                                                    pass_type = geometry_type
                                                    vlm_conf = geometry_conf
                                            
                                            else:
                                                # Unknown VLM response - use geometry
                                                pass_type = geometry_type
                                                vlm_conf = geometry_conf
                                
                                # Determine result
                                if passer_team == receiver_team and passer_team != "Unknown":
                                    result = "Success"
                                elif passer_team != receiver_team and passer_team != "Unknown" and receiver_team != "Unknown":
                                    result = "Fail"
                                else:
                                    result = "Unknown"
                                
                                # Normalize pass type
                                if "throw" in pass_type.lower() and "short" not in pass_type.lower() and "long" not in pass_type.lower():
                                    pass_type = "Short throw-in" if distance < SHORT_PASS_THRESHOLD else "Long throw-in"
                                
                                if pass_type not in pass_types:
                                    pass_type = "Short pass" if distance < SHORT_PASS_THRESHOLD else "Long pass"
                                
                                # Create pass event
                                pass_event = {
                                    "time": format_time(f_idx / fps),
                                    "frame": f_idx,
                                    "from_player": int(current_owner),
                                    "to_player": int(tid),
                                    "from_team": passer_team,
                                    "to_team": receiver_team,
                                    "pass_type": pass_type,
                                    "result": result,
                                    "distance_px": round(distance, 1),
                                    "confidence": vlm_conf
                                }
                                
                                pass_events.append(pass_event)
                                recent_pass = pass_event
                                last_pass_frame = f_idx
                                
                                # Update stats
                                if passer_team in pass_stats and pass_type in pass_stats[passer_team]:
                                    pass_stats[passer_team][pass_type]["total"] += 1
                                    if result == "Success":
                                        pass_stats[passer_team][pass_type]["success"] += 1
                                    elif result == "Fail":
                                        pass_stats[passer_team][pass_type]["fail"] += 1
                                
                                if debug:
                                    print(f"  [Pass] {pass_event['time']} | {passer_team} #{current_owner} → {receiver_team} #{tid} | {pass_type} | {result} | Conf: {vlm_conf}%")
                
                current_owner = tid
                ownership_start_frame = f_idx
            
            # === SHOT DETECTION ===
            # Check if player with ball might be taking a shot
            if current_owner is not None and ball_xy is not None:
                shooter_pos = player_positions.get(current_owner)
                shooter_bbox = player_bboxes.get(current_owner)
                shooter_team = player_teams.get(current_owner, "Unknown")
                
                if shooter_pos is not None and shooter_team not in ["Unknown", "Referee"]:
                    is_shot, shot_type, shot_conf = shot_detector.detect_shot(
                        f_idx, frame, ball_xy, current_owner, shooter_pos,
                        shooter_bbox, shooter_team
                    )
                    
                    if is_shot:
                        shot_event = shot_detector.record_shot(
                            f_idx, fps, current_owner, shooter_team, shot_type,
                            shot_conf, shooter_pos, ball_xy
                        )
                        recent_shot = shot_event
                        
                        if debug:
                            print(f"  [Shot] {shot_event['time']} | {shooter_team} #{current_owner} | {shot_type} | Conf: {shot_conf}%")
        
        # === ANNOTATE FRAME ===
        if SAVE_ANNOTATED_VIDEO:
            annotated_frame = annotate_frame(frame, recent_pass, recent_shot, None, ball_xy)
            video_writer.write(annotated_frame)
    
    # Close video writer
    if video_writer:
        video_writer.release()
        print(f"\n🎥 Annotated video saved to: {output_video_path}")
    
    # Get shot events and stats
    shot_events = shot_detector.get_shot_events()
    shot_stats = shot_detector.get_stats()
    
    # Save player images
    print("\n" + "=" * 80)
    print("📸 SAVING PLAYER IMAGES")
    print("=" * 80)
    saved_count = player_reid.save_player_images(output_dir="player_images")
    print(f"✅ Saved {saved_count} player images with bounding boxes")
    print(f"📁 Location: player_images/")
    print(f"📋 Format: player_01_Blue.jpg, player_02_Red.jpg, etc.")
    
    # === PRINT SUMMARY ===
    print("\n" + "=" * 80)
    print("📊 PASS ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Pass Type':<20} | {'Blue Team':^12} | {'Red Team':^12} | {'Total':^12}")
    print("-" * 65)
    
    for pt in pass_types:
        blue = pass_stats[TEAM_A_NAME][pt]['total']
        red = pass_stats[TEAM_B_NAME][pt]['total']
        total = blue + red
        print(f"{pt:<20} | {blue:^12} | {red:^12} | {total:^12}")
    
    total_blue = sum(pass_stats[TEAM_A_NAME][pt]['total'] for pt in pass_types)
    total_red = sum(pass_stats[TEAM_B_NAME][pt]['total'] for pt in pass_types)
    print("-" * 65)
    print(f"{'TOTAL':<20} | {total_blue:^12} | {total_red:^12} | {len(pass_events):^12}")
    
    # Print shot summary
    shot_detector.print_summary()
    
    # === SAVE REPORTS ===
    print("\n" + "=" * 80)
    print("💾 SAVING REPORTS")
    print("=" * 80)
    
    # CSV reports
    video_name = os.path.splitext(os.path.basename(video_path))[0].replace(' ', '_')
    
    csv_path = f"{video_name}_pass_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'frame', 'from_player', 'to_player', 
                                                'from_team', 'to_team', 'pass_type', 'result', 
                                                'distance_px', 'confidence'])
        writer.writeheader()
        for event in pass_events:
            writer.writerow(event)
    print(f"✅ Pass CSV: {csv_path}")
    
    # Shot CSV
    shot_csv_path = f"{video_name}_shot_report.csv"
    generate_shot_report_csv(shot_events, shot_csv_path)
    
    # JSON reports - convert numpy types to Python types
    def convert_to_json_serializable(obj):
        """Convert numpy types to JSON serializable Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_path = f"{video_name}_pass_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json_data = convert_to_json_serializable({"passes": pass_events, "stats": pass_stats})
        json.dump(json_data, f, indent=2)
    print(f"✅ Pass JSON: {json_path}")
    
    shot_json_path = f"{video_name}_shot_report.json"
    generate_shot_report_json(shot_events, shot_stats, shot_json_path)
    
    # HTML Report with annotated video popup
    html_path = generate_full_scout_report_html(
        pass_events, 
        shot_events, 
        video_path, 
        output_video_path,
        pass_stats, 
        shot_stats, 
        fps
    )
    print(f"✅ HTML Report: {html_path}")
    
    # Instructions
    print("\n" + "=" * 80)
    print("🎬 VIDEO PLAYBACK INSTRUCTIONS")
    print("=" * 80)
    print(f"\nTo enable video playback in the HTML report:")
    print(f"1. Copy BOTH files to the SAME folder:")
    print(f"   - {os.path.basename(video_path)} (original video)")
    print(f"   - {output_video_path} (annotated video - used in popup!)")
    print(f"   - {html_path}")
    print(f"\n2. Open {html_path} in Chrome/Edge/Firefox")
    print(f"\n3. Click any pass or shot to watch the ANNOTATED VIDEO!")
    print("=" * 80)
    
    return pass_events, shot_events, pass_stats, shot_stats


# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScoutMe Soccer Analysis v6")
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    try:
        run_analysis(args.source, args.debug)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()




