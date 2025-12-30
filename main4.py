import os
import cv2
import torch
import csv
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


# --- 1. CONFIG ---
DATA_DIR = 'data'
PLAYER_MODEL = os.path.join(DATA_DIR, 'football-player-detection.pt')
BALL_MODEL = os.path.join(DATA_DIR, 'football-ball-detection.pt')
PITCH_MODEL = os.path.join(DATA_DIR, 'football-pitch-detection.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Team colors (configurable)
TEAM_A_NAME = "Blue"
TEAM_B_NAME = "Red"

# === TUNED DETECTION PARAMETERS ===
BALL_PROXIMITY_THRESHOLD = 70      # pixels - balanced
EVENT_COOLDOWN_FRAMES = 20         # frames - balanced for ~40 passes
MIN_PASS_DISTANCE = 35             # minimum pixels to count as pass
MIN_OWNERSHIP_FRAMES = 6           # player must have ball for X frames before pass counts

# Field zones (as percentage of frame width/height)
SIDELINE_MARGIN = 0.04             # 4% from edge = sideline area (for throw-ins)
CROSS_ZONE_WIDTH = 0.25            # 25% from sides = wide areas
HEADER_HEIGHT_RATIO = 0.22         # ball in top 22% of player bbox = header

# Pass classification thresholds  
SHORT_PASS_THRESHOLD = 100         # pixels - balanced for short/long detection
LONG_PASS_THRESHOLD = 280          # pixels

# Video annotation settings
SAVE_ANNOTATED_VIDEO = True        # Save video with pass annotations

# VLM settings
USE_VLM_FOR_VALIDATION = False     # DISABLED - geometry is more reliable for validation
USE_VLM_FOR_CLASSIFICATION = True  # Use VLM only for pass type classification


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


def vlm_query(frame_crop, prompt, max_tokens=10):
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
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).lower()
        return response
    except Exception as e:
        return "unknown"


def classify_pass_type_vlm(frame_crop, geometry_suggestion):
    """
    Use VLM to classify the type of pass/action.
    Geometry suggestion helps guide the VLM with context.
    """
    prompt = f"""Analyze this soccer match frame carefully.

CONTEXT: A pass/action just occurred. The geometry analysis suggests this is a "{geometry_suggestion}".

Look at:
- Player body posture and stance
- Ball position relative to players
- Whether any player is near the sideline (edge of pitch)
- Whether the ball appears to be aerial or on ground
- Player arm positions (raised arms = throw-in)

What type of action is this? Choose ONE:

(A) SHORT PASS - Ground pass, ball travels short distance (<15 meters), player kicks with inside of foot
(B) LONG PASS - Ground pass, ball travels longer distance (>15 meters), player uses more power
(C) CROSS - Ball sent from wide/wing area into the penalty box, usually aerial, aimed at attackers
(D) THROW-IN - Player standing at sideline, holding ball with BOTH HANDS above head, throwing it in
(E) HEADER - Player making contact with the ball using their HEAD, ball is in the air

Answer with ONLY the letter (A, B, C, D, or E):"""
    
    response = vlm_query(frame_crop, prompt, max_tokens=5)
    
    response_upper = response.upper().strip()
    
    # Map response to pass type
    if 'D' in response_upper or 'THROW' in response_upper:
        return "Throw-in"
    elif 'E' in response_upper or 'HEAD' in response_upper:
        return "Header"
    elif 'C' in response_upper or 'CROSS' in response_upper:
        return "Cross"
    elif 'B' in response_upper or 'LONG' in response_upper:
        return "Long pass"
    elif 'A' in response_upper or 'SHORT' in response_upper:
        return "Short pass"
    else:
        # Return geometry suggestion as fallback
        return geometry_suggestion


def is_header_action(ball_xy, player_bbox, ball_history=None):
    """
    Check if ball position suggests a header (ball near head level).
    VERY STRICT: requires ball to be clearly at head height AND showing aerial characteristics.
    """
    x1, y1, x2, y2 = player_bbox
    player_height = y2 - y1
    head_zone_bottom = y1 + (player_height * HEADER_HEIGHT_RATIO)
    
    # Basic check: ball must be at head height (top 18% of player bbox)
    ball_at_head = ball_xy[1] < head_zone_bottom
    
    if not ball_at_head:
        return False
    
    # Additional check 1: ball should show vertical movement (aerial ball)
    if ball_history and len(ball_history) >= 4:
        recent_y = [pos[1][1] for pos in ball_history[-4:]]
        y_variance = max(recent_y) - min(recent_y)
        # Ball MUST show significant vertical movement for a header (min 40px)
        if y_variance < 40:
            return False
        
        # Additional check 2: ball should be moving (not stationary at head)
        recent_x = [pos[1][0] for pos in ball_history[-4:]]
        x_variance = max(recent_x) - min(recent_x)
        total_movement = y_variance + x_variance
        if total_movement < 60:  # Ball must be moving significantly
            return False
    else:
        # Without enough history, be very conservative - don't classify as header
        return False
    
    return True


def is_sideline_position(x_pos, y_pos, frame_width, frame_height):
    """
    Check if position is near sideline (for throw-in detection).
    Sidelines can be on LEFT/RIGHT edges OR TOP/BOTTOM edges depending on camera angle.
    """
    # Check horizontal sidelines (left/right edges)
    left_margin = frame_width * SIDELINE_MARGIN
    right_margin = frame_width * (1 - SIDELINE_MARGIN)
    near_horizontal_sideline = x_pos < left_margin or x_pos > right_margin
    
    # Check vertical sidelines (top/bottom edges) - for side-view cameras
    top_margin = frame_height * SIDELINE_MARGIN
    bottom_margin = frame_height * (1 - SIDELINE_MARGIN)
    near_vertical_sideline = y_pos < top_margin or y_pos > bottom_margin
    
    return near_horizontal_sideline or near_vertical_sideline


def is_wide_position(x_pos, frame_width):
    """Check if position is in wide area (for cross detection)"""
    left_zone = frame_width * CROSS_ZONE_WIDTH
    right_zone = frame_width * (1 - CROSS_ZONE_WIDTH)
    return x_pos < left_zone or x_pos > right_zone


def is_central_position(x_pos, frame_width):
    """Check if position is in central/penalty box area"""
    left_zone = frame_width * CROSS_ZONE_WIDTH
    right_zone = frame_width * (1 - CROSS_ZONE_WIDTH)
    return left_zone <= x_pos <= right_zone


def is_referee(frame, bbox):
    """
    Check if detected person is likely a referee (black/yellow/green kit).
    Returns True if referee, False if player.
    """
    x1, y1, x2, y2 = map(int, bbox)
    jersey_y2 = y1 + int((y2 - y1) * 0.5)
    crop = frame[max(0, y1):jersey_y2, max(0, x1):x2]
    
    if crop.size == 0:
        return False
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Black kit (referee): Low saturation, low value
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 50, 80])
    
    # Yellow kit (referee): Hue ~20-35
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    
    # Bright green/lime (referee): Hue ~35-85
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    
    black_mask = cv2.inRange(hsv, black_lower, black_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    total_pixels = crop.shape[0] * crop.shape[1]
    black_pct = cv2.countNonZero(black_mask) / total_pixels if total_pixels > 0 else 0
    yellow_pct = cv2.countNonZero(yellow_mask) / total_pixels if total_pixels > 0 else 0
    green_pct = cv2.countNonZero(green_mask) / total_pixels if total_pixels > 0 else 0
    
    # LESS STRICT: Only flag as referee if VERY clearly referee colors
    # >50% black (not just shadowed), or >35% bright yellow/green
    if black_pct > 0.50 or yellow_pct > 0.35 or green_pct > 0.35:
        return True
    
    return False


def detect_team_color_opencv(frame, bbox, team_a_name="Blue", team_b_name="Red"):
    """
    Detect jersey color using OpenCV color analysis.
    Enhanced for better detection. Also filters referees.
    """
    # First check if this is a referee
    if is_referee(frame, bbox):
        return "Referee"
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Get the upper body (jersey area) - top 60% of bounding box for better coverage
    jersey_y1 = y1
    jersey_y2 = y1 + int((y2 - y1) * 0.6)
    
    # Add some horizontal padding
    pad_x = int((x2 - x1) * 0.1)
    crop = frame[max(0, jersey_y1):jersey_y2, max(0, x1+pad_x):max(0, x2-pad_x)]
    
    if crop.size == 0:
        return "Unknown"
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Blue jersey: Hue ~100-130 (expanded range)
    blue_lower = np.array([85, 40, 40])
    blue_upper = np.array([135, 255, 255])
    
    # Red jersey: Hue ~0-15 and 165-180 (expanded range)
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    # Create masks
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Count pixels
    blue_pixels = cv2.countNonZero(blue_mask)
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = crop.shape[0] * crop.shape[1]
    
    # Calculate percentages
    blue_pct = blue_pixels / total_pixels if total_pixels > 0 else 0
    red_pct = red_pixels / total_pixels if total_pixels > 0 else 0
    
    min_pct = 0.05  # At least 5% of pixels should be team color
    
    if blue_pct > red_pct and blue_pct > min_pct:
        return team_a_name
    elif red_pct > blue_pct and red_pct > min_pct:
        return team_b_name
    else:
        # Fallback: check average BGR values
        avg_color = np.mean(crop, axis=(0, 1))
        b, g, r = avg_color
        
        # Blue dominant
        if b > r * 1.15 and b > g * 1.1:
            return team_a_name
        # Red dominant
        elif r > b * 1.15 and r > g * 0.9:
            return team_b_name
        else:
            return "Unknown"


def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def classify_pass_by_geometry(distance, passer_pos, receiver_pos, ball_xy, passer_bbox, frame_width, frame_height, ball_history=None, pitch_bounds=None):
    """
    Classify pass type using geometric analysis.
    Uses pitch detection when available for more accurate zone detection.
    Returns: 'Short pass', 'Long pass', 'Cross', 'Short throw-in', 'Long throw-in', 'Header'
    """
    # Check for header first (ball at head height with aerial characteristics)
    if is_header_action(ball_xy, passer_bbox, ball_history):
        return "Header"
    
    # Check for throw-in (passer near sideline)
    # Use pitch detection if available, otherwise use frame-based detection
    near_sideline = False
    if pitch_bounds and pitch_bounds.get('detected', False):
        # Use 8% margin for throw-in detection (more lenient)
        near_sideline = is_near_sideline_pitch(passer_pos[0], passer_pos[1], pitch_bounds, margin_pct=0.08)
    else:
        near_sideline = is_sideline_position(passer_pos[0], passer_pos[1], frame_width, frame_height)
    
    if near_sideline:
        if distance < SHORT_PASS_THRESHOLD:
            return "Short throw-in"
        else:
            return "Long throw-in"
    
    # Check for cross (pass from wide area to central area)
    # Use pitch detection if available
    if pitch_bounds and pitch_bounds.get('detected', False):
        passer_wide = is_in_wide_area_pitch(passer_pos[0], pitch_bounds, wide_pct=0.25)
        receiver_central = is_in_central_area_pitch(receiver_pos[0], pitch_bounds, central_pct=0.50)
    else:
        passer_wide = is_wide_position(passer_pos[0], frame_width)
        receiver_central = is_central_position(receiver_pos[0], frame_width)
    
    if passer_wide and receiver_central and distance > SHORT_PASS_THRESHOLD:
        return "Cross"
    
    # Regular pass classification by distance
    if distance < SHORT_PASS_THRESHOLD:
        return "Short pass"
    else:
        return "Long pass"


# --- 3. PITCH DETECTION HELPER ---
def detect_pitch_boundaries(pitch_model, frame):
    """
    Detect pitch boundaries using the pitch detection model.
    Returns dict with sideline positions if detected.
    """
    try:
        results = pitch_model(frame, imgsz=1280, verbose=False)[0]
        
        if results.masks is not None and len(results.masks) > 0:
            # Get the largest mask (should be the pitch)
            mask = results.masks.data[0].cpu().numpy()
            
            # Find bounding box of the pitch
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
        
        # Try bounding boxes if no masks
        if results.boxes is not None and len(results.boxes) > 0:
            # Get largest detection (pitch)
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
    except Exception as e:
        pass
    
    return {'detected': False}


def is_near_sideline_pitch(pos_x, pos_y, pitch_bounds, margin_pct=0.05):
    """
    Check if position is near sideline using actual pitch boundaries.
    More accurate than frame-based detection.
    """
    if not pitch_bounds.get('detected', False):
        return False
    
    left = pitch_bounds['left_sideline']
    right = pitch_bounds['right_sideline']
    top = pitch_bounds['top_sideline']
    bottom = pitch_bounds['bottom_sideline']
    
    pitch_width = right - left
    pitch_height = bottom - top
    
    margin_x = pitch_width * margin_pct
    margin_y = pitch_height * margin_pct
    
    # Check if near any sideline
    near_left = pos_x < (left + margin_x)
    near_right = pos_x > (right - margin_x)
    near_top = pos_y < (top + margin_y)
    near_bottom = pos_y > (bottom - margin_y)
    
    return near_left or near_right or near_top or near_bottom


def is_in_wide_area_pitch(pos_x, pitch_bounds, wide_pct=0.25):
    """
    Check if position is in wide area using pitch boundaries.
    """
    if not pitch_bounds.get('detected', False):
        return False
    
    left = pitch_bounds['left_sideline']
    right = pitch_bounds['right_sideline']
    pitch_width = right - left
    
    wide_margin = pitch_width * wide_pct
    
    return pos_x < (left + wide_margin) or pos_x > (right - wide_margin)


def is_in_central_area_pitch(pos_x, pitch_bounds, central_pct=0.50):
    """
    Check if position is in central area using pitch boundaries.
    """
    if not pitch_bounds.get('detected', False):
        return False
    
    left = pitch_bounds['left_sideline']
    right = pitch_bounds['right_sideline']
    pitch_width = right - left
    center = (left + right) / 2
    
    central_margin = pitch_width * central_pct / 2
    
    return (center - central_margin) < pos_x < (center + central_margin)


# --- 4. MAIN ENGINE ---
def run_analysis(video_path, use_vlm_classification=True, debug=False):
    v_info = sv.VideoInfo.from_video_path(video_path)
    frame_width = v_info.width
    frame_height = v_info.height
    fps = v_info.fps
    
    print()
    print("=" * 60)
    print("🚀 LOADING ALL 4 AI MODELS")
    print("=" * 60)
    
    # Model 1: Player Detection
    print(f"[1/4] Loading Player Detection Model...")
    p_m = YOLO(PLAYER_MODEL).to(DEVICE)
    print(f"      ✅ Player Model: {PLAYER_MODEL}")
    
    # Model 2: Ball Detection
    print(f"[2/4] Loading Ball Detection Model...")
    b_m = YOLO(BALL_MODEL).to(DEVICE)
    print(f"      ✅ Ball Model: {BALL_MODEL}")
    
    # Model 3: Pitch Detection
    print(f"[3/4] Loading Pitch Detection Model...")
    pitch_m = None
    use_pitch_detection = False
    try:
        if os.path.exists(PITCH_MODEL):
            pitch_m = YOLO(PITCH_MODEL).to(DEVICE)
            use_pitch_detection = True
            print(f"      ✅ Pitch Model: {PITCH_MODEL}")
        else:
            print(f"      ⚠️  Pitch model not found: {PITCH_MODEL}")
    except Exception as e:
        print(f"      ⚠️  Could not load pitch model: {e}")
    
    # Model 4: Molmo VLM (already loaded at startup)
    print(f"[4/4] Molmo-7B VLM...")
    print(f"      ✅ VLM Model: allenai/Molmo-7B-D-0924 (loaded at startup)")
    
    print("=" * 60)
    print("✅ ALL 4 MODELS READY!")
    print("=" * 60)
    print()
    
    tracker = sv.ByteTrack()
    
    pass_events = []
    player_teams = {}
    player_positions = {}
    player_bboxes = {}
    current_owner = None
    ownership_start_frame = 0  # Track when current owner got the ball
    last_event_frame = -100
    
    # Ball trajectory tracking for header detection
    ball_history = []  # Store recent ball positions
    BALL_HISTORY_SIZE = 10
    
    # Pitch detection cache
    pitch_bounds = {'detected': False}
    PITCH_DETECT_INTERVAL = 100  # Re-detect pitch every N frames
    
    # Debug counters
    ownership_changes = 0
    filtered_by_distance = 0
    filtered_by_cooldown = 0
    filtered_by_ownership_duration = 0
    filtered_by_referee = 0

    # Enhanced stats structure
    pass_types = ["Short pass", "Long pass", "Cross", "Short throw-in", "Long throw-in", "Header"]
    stats = {
        TEAM_A_NAME: {pt: {"success": 0, "fail": 0, "total": 0} for pt in pass_types},
        TEAM_B_NAME: {pt: {"success": 0, "fail": 0, "total": 0} for pt in pass_types}
    }

    print(f"🎬 Processing: {video_path}...")
    print(f"📐 Frame size: {frame_width}x{frame_height} @ {fps:.1f}fps")
    print()
    print("📋 MODEL STATUS:")
    print(f"   🏃 Player Detection:  ✅ Active (YOLO)")
    print(f"   ⚽ Ball Detection:    ✅ Active (YOLO)")
    print(f"   🏟️  Pitch Detection:   {'✅ Active (YOLO)' if use_pitch_detection else '⚠️ Disabled'}")
    print(f"   🧠 Pass Classification: {'✅ Active (Molmo-7B VLM)' if use_vlm_classification else '⚠️ Geometry Only'}")
    print(f"   👕 Team Detection:    ✅ Active (OpenCV HSV)")
    print()
    print(f"⚙️  Settings: Ball proximity={BALL_PROXIMITY_THRESHOLD}px, Cooldown={EVENT_COOLDOWN_FRAMES} frames")
    print(f"🎥 Video Annotation: {'✅ Enabled' if SAVE_ANNOTATED_VIDEO else '❌ Disabled'}")
    print()
    
    # Model usage counters
    model_stats = {
        'player_detections': 0,
        'ball_detections': 0,
        'pitch_detections': 0,
        'vlm_classifications': 0
    }
    
    # Setup video writer for annotated output
    video_writer = None
    output_video_path = 'annotated_passes.mp4'
    if SAVE_ANNOTATED_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Annotation helpers
    recent_passes = []  # Store recent pass events for display
    PASS_DISPLAY_DURATION = 60  # frames to show pass annotation
    
    for f_idx, frame in enumerate(tqdm(sv.get_video_frames_generator(video_path), total=v_info.total_frames)):
        
        # Detect pitch boundaries periodically
        if use_pitch_detection and (f_idx % PITCH_DETECT_INTERVAL == 0):
            pitch_bounds = detect_pitch_boundaries(pitch_m, frame)
            model_stats['pitch_detections'] += 1
            if pitch_bounds['detected'] and f_idx == 0:
                print(f"📐 Pitch detected: {pitch_bounds['pitch_width']}x{pitch_bounds['pitch_height']}px")
        
        # Detection - Player Model
        p_det = tracker.update_with_detections(sv.Detections.from_ultralytics(p_m(frame, imgsz=1280, verbose=False)[0]))
        model_stats['player_detections'] += 1
        
        # Detection - Ball Model
        b_det = sv.Detections.from_ultralytics(b_m(frame, imgsz=640, verbose=False)[0])
        model_stats['ball_detections'] += 1
        
        # Update player positions and bboxes
        if p_det.tracker_id is not None:
            for tid, p_xy, bbox in zip(p_det.tracker_id, p_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER), p_det.xyxy):
                player_positions[tid] = p_xy
                player_bboxes[tid] = bbox
                
                # Detect team color for all players (re-detect periodically)
                if tid not in player_teams or (f_idx % 300 == 0):
                    detected_team = detect_team_color_opencv(frame, bbox, TEAM_A_NAME, TEAM_B_NAME)
                    if detected_team != "Unknown" or tid not in player_teams:
                        player_teams[tid] = detected_team
        
        # Ball detection
        ball_coords = b_det.get_anchors_coordinates(sv.Position.CENTER)
        ball_xy = ball_coords[0] if len(ball_coords) > 0 else None
        
        # Track ball history for trajectory analysis
        if ball_xy is not None:
            ball_history.append((f_idx, ball_xy.copy()))
            if len(ball_history) > BALL_HISTORY_SIZE:
                ball_history.pop(0)
        
        if ball_xy is not None and p_det.tracker_id is not None:
            # Find closest player to ball
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
            
            # Check if ball is close enough to a player
            if min_dist < BALL_PROXIMITY_THRESHOLD and closest_player is not None:
                tid = closest_player
                bbox = closest_bbox
                p_xy = closest_pos
                
                # Ownership change detected
                if current_owner is not None and tid != current_owner:
                    ownership_changes += 1
                    
                    # Check cooldown
                    if f_idx - last_event_frame <= EVENT_COOLDOWN_FRAMES:
                        filtered_by_cooldown += 1
                    # Check minimum ownership duration (previous owner must have had ball long enough)
                    elif f_idx - ownership_start_frame < MIN_OWNERSHIP_FRAMES:
                        filtered_by_ownership_duration += 1
                    else:
                        # Calculate pass distance
                        passer_pos = player_positions.get(current_owner, p_xy)
                        receiver_pos = p_xy
                        distance = np.linalg.norm(passer_pos - receiver_pos)
                        
                        # Filter very short distances (noise)
                        if distance < MIN_PASS_DISTANCE:
                            filtered_by_distance += 1
                        else:
                            # Get team colors
                            passer_team = player_teams.get(current_owner, "Unknown")
                            receiver_team = player_teams.get(tid, "Unknown")
                            
                            # Re-detect if unknown
                            if passer_team == "Unknown" and current_owner in player_bboxes:
                                passer_team = detect_team_color_opencv(frame, player_bboxes[current_owner], TEAM_A_NAME, TEAM_B_NAME)
                                player_teams[current_owner] = passer_team
                            
                            if receiver_team == "Unknown":
                                receiver_team = detect_team_color_opencv(frame, bbox, TEAM_A_NAME, TEAM_B_NAME)
                                player_teams[tid] = receiver_team
                            
                            # Get passer bbox for header detection
                            passer_bbox = player_bboxes.get(current_owner, bbox)
                            
                            # CLASSIFY PASS TYPE - Geometry first (with pitch detection if available)
                            geometry_pass_type = classify_pass_by_geometry(
                                distance, passer_pos, receiver_pos, ball_xy,
                                passer_bbox, frame_width, frame_height, ball_history, pitch_bounds
                            )
                            
                            # Use VLM for classification if enabled
                            if use_vlm_classification and USE_VLM_FOR_CLASSIFICATION:
                                # Get context crop for VLM
                                crop_size = 250
                                crop = frame[
                                    max(0, int(ball_xy[1])-crop_size):min(frame_height, int(ball_xy[1])+crop_size),
                                    max(0, int(ball_xy[0])-crop_size):min(frame_width, int(ball_xy[0])+crop_size)
                                ]
                                
                                if crop.size > 0:
                                    pass_type = classify_pass_type_vlm(crop, geometry_pass_type)
                                    model_stats['vlm_classifications'] += 1
                                else:
                                    pass_type = geometry_pass_type
                            else:
                                pass_type = geometry_pass_type
                            
                            # Normalize throw-in type
                            if "throw" in pass_type.lower():
                                if "short" not in pass_type.lower() and "long" not in pass_type.lower():
                                    pass_type = "Long throw-in" if distance > SHORT_PASS_THRESHOLD else "Short throw-in"
                            
                            # Ensure pass_type is valid
                            if pass_type not in pass_types:
                                pass_type = geometry_pass_type
                            
                            # Skip if either player is a referee
                            if passer_team == "Referee" or receiver_team == "Referee":
                                filtered_by_referee += 1
                                current_owner = tid
                                continue
                            
                            # Determine success/failure
                            if passer_team == receiver_team and passer_team != "Unknown":
                                result = "Success"
                            elif passer_team != receiver_team and passer_team != "Unknown" and receiver_team != "Unknown":
                                result = "Fail"
                            else:
                                result = "Unknown"
                            
                            # Update stats (only for known teams)
                            if passer_team in stats and pass_type in stats[passer_team]:
                                stats[passer_team][pass_type]["total"] += 1
                                if result == "Success":
                                    stats[passer_team][pass_type]["success"] += 1
                                elif result == "Fail":
                                    stats[passer_team][pass_type]["fail"] += 1
                            
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
                                "passer_x": round(passer_pos[0], 1),
                                "passer_y": round(passer_pos[1], 1)
                            })
                            
                            if debug:
                                print(f"  [Pass] {format_time(f_idx/fps)} | {passer_team} #{current_owner} → {receiver_team} #{tid} | {pass_type} | {result}")
                            
                            # Store pass for video annotation
                            if SAVE_ANNOTATED_VIDEO:
                                recent_passes.append({
                                    'frame': f_idx,
                                    'passer_pos': passer_pos.copy(),
                                    'receiver_pos': receiver_pos.copy(),
                                    'pass_type': pass_type,
                                    'result': result,
                                    'passer_team': passer_team
                                })
                            
                            last_event_frame = f_idx
                
                if current_owner != tid:
                    ownership_start_frame = f_idx
                current_owner = tid
        
        # --- VIDEO ANNOTATION ---
        if SAVE_ANNOTATED_VIDEO and video_writer is not None:
            annotated_frame = frame.copy()
            
            # Draw player detections with team colors
            if p_det.tracker_id is not None:
                for tid_draw, bbox_draw in zip(p_det.tracker_id, p_det.xyxy):
                    team = player_teams.get(tid_draw, "Unknown")
                    if team == TEAM_A_NAME:
                        color = (255, 0, 0)  # Blue
                    elif team == TEAM_B_NAME:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (128, 128, 128)  # Gray for unknown
                    
                    x1, y1, x2, y2 = map(int, bbox_draw)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"#{tid_draw}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball
            if ball_xy is not None:
                cv2.circle(annotated_frame, (int(ball_xy[0]), int(ball_xy[1])), 10, (0, 255, 255), -1)
            
            # Draw recent passes (arrows)
            recent_passes = [p for p in recent_passes if f_idx - p['frame'] < PASS_DISPLAY_DURATION]
            for pass_info in recent_passes:
                p1 = (int(pass_info['passer_pos'][0]), int(pass_info['passer_pos'][1]))
                p2 = (int(pass_info['receiver_pos'][0]), int(pass_info['receiver_pos'][1]))
                
                # Color based on result
                if pass_info['result'] == "Success":
                    arrow_color = (0, 255, 0)  # Green
                else:
                    arrow_color = (0, 0, 255)  # Red
                
                cv2.arrowedLine(annotated_frame, p1, p2, arrow_color, 3, tipLength=0.1)
                
                # Pass label
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                label = f"{pass_info['pass_type']}"
                cv2.putText(annotated_frame, label, (mid_x, mid_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
            
            # Frame info overlay
            cv2.putText(annotated_frame, f"Frame: {f_idx} | Time: {format_time(f_idx/fps)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Passes: {len(pass_events)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            video_writer.write(annotated_frame)
        
        # VRAM management
        if f_idx % 150 == 0 and f_idx > 0:
            torch.cuda.empty_cache()

    # Close video writer
    if SAVE_ANNOTATED_VIDEO and video_writer is not None:
        video_writer.release()
        print(f"\n🎥 Annotated video saved to: {output_video_path}")
    
    # Debug info
    print()
    print("=" * 80)
    print("🔍 DEBUG INFO")
    print("=" * 80)
    print(f"Total ownership changes detected: {ownership_changes}")
    print(f"Filtered by cooldown: {filtered_by_cooldown}")
    print(f"Filtered by ownership duration: {filtered_by_ownership_duration}")
    print(f"Filtered by min distance: {filtered_by_distance}")
    print(f"Final passes recorded: {len(pass_events)}")
    
    print()
    print("=" * 80)
    print("🤖 MODEL USAGE STATISTICS")
    print("=" * 80)
    print(f"🏃 Player Detection Model:  {model_stats['player_detections']:,} inferences")
    print(f"⚽ Ball Detection Model:    {model_stats['ball_detections']:,} inferences")
    print(f"🏟️  Pitch Detection Model:   {model_stats['pitch_detections']:,} inferences")
    print(f"🧠 Molmo-7B VLM:            {model_stats['vlm_classifications']:,} classifications")
    
    # Team detection stats
    team_counts = {"Blue": 0, "Red": 0, "Unknown": 0}
    for team in player_teams.values():
        if team in team_counts:
            team_counts[team] += 1
        else:
            team_counts["Unknown"] += 1
    print(f"Players detected - Blue: {team_counts['Blue']}, Red: {team_counts['Red']}, Unknown: {team_counts['Unknown']}")

    # Save detailed CSV
    csv_path = 'detailed_pass_report.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["time", "frame", "from_player", "to_player", "from_team", "to_team", 
                      "pass_type", "result", "distance_px", "passer_x", "passer_y"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pass_events)
    
    # Print summary matching your manual analysis format
    print()
    print("=" * 80)
    print("📊 PASS ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    # Create table header
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
    
    # Team totals
    for team_name in [TEAM_A_NAME, TEAM_B_NAME]:
        total = sum(stats[team_name][pt]["total"] for pt in pass_types)
        success = sum(stats[team_name][pt]["success"] for pt in pass_types)
        fail = sum(stats[team_name][pt]["fail"] for pt in pass_types)
        rate = (success / (success + fail) * 100) if (success + fail) > 0 else 0
        print(f"\n🏃 {team_name} Team Total: {total} passes, {success} success, {fail} fail ({rate:.1f}% success rate)")
    
    print()
    print("=" * 80)
    print(f"✅ Detailed report saved to: {csv_path}")
    print("=" * 80)
    
    return stats, pass_events


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Soccer Pass Analysis with VLM')
    parser.add_argument('--source', type=str, default='input_video/video_segment.mp4', help='Path to video file')
    parser.add_argument('--no-vlm', action='store_true', help='Disable VLM classification (geometry only)')
    parser.add_argument('--debug', action='store_true', help='Show detailed pass detection logs')
    args = parser.parse_args()
    
    run_analysis(args.source, use_vlm_classification=not args.no_vlm, debug=args.debug)
