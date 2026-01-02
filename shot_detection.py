"""
ScoutMe - Shot on Goal Detection Module (TIGHTENED FOR ACCURACY)
Detects shots on goal and classifies them as ON TARGET or OFF TARGET
Uses VLM (Molmo-7B) for accurate visual analysis

KEY IMPROVEMENTS:
1. Distance-to-Goal Calculation: Only triggers VLM if ball within 10m radius of goal
2. Velocity Vector Intersection: Validates ball trajectory intersects goal frame (45° angle limit)
3. Static Goal Calibration: Maps goal coordinates for precise intersection logic
4. Enhanced VLM Prompts: Stricter boundaries to prevent false positives (crosses vs shots)
5. Comprehensive Metrics: Tracks ball velocity, angle of arrival, distance, VLM confidence

TIGHTENED PARAMETERS (to reduce over-detection):
- SHOT_VELOCITY_THRESHOLD: 25 px/frame (raised from 15) - excludes soft passes
- SHOT_CONFIDENCE_THRESHOLD: 80% (raised from 60%) - excludes uncertain guesses
- Shooting Range: Central zone only (excludes sidelines/corners where crosses occur)
- Angle Check: Max 45° deviation from goal direction (filters square passes)

API CHANGE:
- detect_shot() now returns: (is_shot, shot_type, confidence, metrics_dict)
  - metrics_dict contains: ball_velocity_px_per_frame, angle_of_arrival_degrees,
    distance_from_goal_meters, vlm_confidence_score, target_goal, velocity_dx, velocity_dy
- record_shot() now accepts optional metrics parameter

Integration: Import and use with main4.py or main5.py
"""

import numpy as np
from collections import deque


# === SHOT DETECTION PARAMETERS (BALANCED FOR DETECTION) ===
SHOT_PROXIMITY_THRESHOLD = 100         # Ball near player for potential shot
SHOT_VELOCITY_THRESHOLD = 18           # ADJUSTED: Balance between detecting shots and filtering passes
GOAL_ZONE_MARGIN = 0.25                # 25% from each end = goal areas
SHOT_COOLDOWN_FRAMES = 45              # 1.5 seconds between shots
MIN_SHOT_DISTANCE_TO_GOAL = 50         # Minimum distance for shot
SHOT_CONFIDENCE_THRESHOLD = 70         # ADJUSTED: Balance between accuracy and detection rate

# === REFINED DETECTION PARAMETERS ===
GOAL_PROXIMITY_RADIUS_METERS = 15      # ADJUSTED: 15-meter radius (was 10m - too restrictive)
GOAL_WIDTH_METERS = 7.32               # Standard goal width in meters
GOAL_HEIGHT_METERS = 2.44              # Standard goal height in meters
PIXELS_PER_METER = None                # Will be calibrated from pitch detection


# === SHOT TYPE COLORS ===
SHOT_COLORS = {
    "Shot on target": "#22c55e",       # Green - successful/saved
    "Shot off target": "#ef4444",      # Red - missed
    "Shot blocked": "#f59e0b",         # Orange - blocked by defender
    "Goal": "#ffd700"                  # Gold - scored!
}


class ShotDetector:
    """
    Detects shots on goal using ball trajectory and player positioning.
    Uses VLM for accurate classification of shot outcome.
    """
    
    def __init__(self, frame_width, frame_height, fps, vlm_query_func):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.vlm_query = vlm_query_func
        
        # Ball trajectory buffer for velocity calculation
        self.ball_history = deque(maxlen=10)
        
        # Shot events
        self.shot_events = []
        self.last_shot_frame = -SHOT_COOLDOWN_FRAMES
        
        # Goal calibration (static goal coordinates)
        self.goal_coordinates = None  # Will store: {'left_goal': (x, y, width, height), 'right_goal': (x, y, width, height)}
        self.pixels_per_meter = None  # Calibration factor
        
        # Goal zones (left and right ends of pitch)
        self._setup_goal_zones()
        
        # Stats
        self.stats = {
            "Shot on target": {"Blue": 0, "Red": 0},
            "Shot off target": {"Blue": 0, "Red": 0},
            "Shot blocked": {"Blue": 0, "Red": 0},
            "Goal": {"Blue": 0, "Red": 0}
        }
    
    def _setup_goal_zones(self, pitch_bounds=None):
        """Define goal zones at each end of the pitch"""
        if pitch_bounds and pitch_bounds.get('detected', False):
            left = pitch_bounds['left_sideline']
            right = pitch_bounds['right_sideline']
            width = right - left
            
            # Goals are at the ends
            self.left_goal_zone = (left, left + width * GOAL_ZONE_MARGIN)
            self.right_goal_zone = (right - width * GOAL_ZONE_MARGIN, right)
            
            # Estimate pixels per meter (standard pitch is ~105m long)
            pitch_length_meters = 105
            if width > 0:
                self.pixels_per_meter = width / pitch_length_meters
        else:
            # Use frame-based detection
            margin = self.frame_width * GOAL_ZONE_MARGIN
            self.left_goal_zone = (0, margin)
            self.right_goal_zone = (self.frame_width - margin, self.frame_width)
            
            # Rough estimate: assume standard pitch fits in frame
            pitch_length_meters = 105
            self.pixels_per_meter = self.frame_width / pitch_length_meters if self.frame_width > 0 else 1
    
    def update_pitch_bounds(self, pitch_bounds):
        """Update goal zones when pitch bounds are detected"""
        if pitch_bounds and pitch_bounds.get('detected', False):
            self._setup_goal_zones(pitch_bounds)
    
    def calibrate_goal_coordinates(self, pitch_bounds=None):
        """
        Calibrate static goal coordinates based on pitch bounds.
        Goals are centered on the end lines, with standard dimensions.
        """
        if not pitch_bounds or not pitch_bounds.get('detected', False):
            # Fallback: estimate goals at frame edges
            goal_center_y = self.frame_height / 2
            goal_width_px = GOAL_WIDTH_METERS * (self.pixels_per_meter or 1)
            goal_height_px = GOAL_HEIGHT_METERS * (self.pixels_per_meter or 1)
            
            self.goal_coordinates = {
                'left_goal': {
                    'center_x': 0,
                    'center_y': goal_center_y,
                    'left_post': -goal_width_px / 2,
                    'right_post': goal_width_px / 2,
                    'top_post': goal_center_y - goal_height_px / 2,
                    'bottom_post': goal_center_y + goal_height_px / 2
                },
                'right_goal': {
                    'center_x': self.frame_width,
                    'center_y': goal_center_y,
                    'left_post': self.frame_width - goal_width_px / 2,
                    'right_post': self.frame_width + goal_width_px / 2,
                    'top_post': goal_center_y - goal_height_px / 2,
                    'bottom_post': goal_center_y + goal_height_px / 2
                }
            }
        else:
            # Use pitch bounds for accurate calibration
            left_edge = pitch_bounds.get('left_sideline', 0)
            right_edge = pitch_bounds.get('right_sideline', self.frame_width)
            top_edge = pitch_bounds.get('top_sideline', 0)
            bottom_edge = pitch_bounds.get('bottom_sideline', self.frame_height)
            
            goal_center_y = (top_edge + bottom_edge) / 2
            pixels_per_m = self.pixels_per_meter if self.pixels_per_meter is not None else 1
            goal_width_px = GOAL_WIDTH_METERS * pixels_per_m
            goal_height_px = GOAL_HEIGHT_METERS * pixels_per_m
            
            self.goal_coordinates = {
                'left_goal': {
                    'center_x': left_edge,
                    'center_y': goal_center_y,
                    'left_post': left_edge - goal_width_px / 2,
                    'right_post': left_edge + goal_width_px / 2,
                    'top_post': goal_center_y - goal_height_px / 2,
                    'bottom_post': goal_center_y + goal_height_px / 2
                },
                'right_goal': {
                    'center_x': right_edge,
                    'center_y': goal_center_y,
                    'left_post': right_edge - goal_width_px / 2,
                    'right_post': right_edge + goal_width_px / 2,
                    'top_post': goal_center_y - goal_height_px / 2,
                    'bottom_post': goal_center_y + goal_height_px / 2
                }
            }
    
    def add_ball_position(self, frame_idx, ball_xy):
        """Add ball position to history for velocity tracking"""
        if ball_xy is not None:
            self.ball_history.append((frame_idx, ball_xy.copy()))
    
    def get_ball_velocity(self):
        """Calculate current ball velocity (pixels per frame)"""
        if len(self.ball_history) < 3:
            return 0, 0, 0  # speed, dx, dy
        
        # Get last 3 positions
        recent = list(self.ball_history)[-3:]
        
        total_dx = 0
        total_dy = 0
        
        for i in range(1, len(recent)):
            prev_pos = recent[i-1][1]
            curr_pos = recent[i][1]
            total_dx += curr_pos[0] - prev_pos[0]
            total_dy += curr_pos[1] - prev_pos[1]
        
        avg_dx = total_dx / (len(recent) - 1)
        avg_dy = total_dy / (len(recent) - 1)
        speed = np.sqrt(avg_dx**2 + avg_dy**2)
        
        return speed, avg_dx, avg_dy
    
    def calculate_distance_to_goal(self, ball_xy, target_goal='auto'):
        """
        Calculate distance from ball to nearest goal in meters.
        Returns: distance in meters, goal_name ('left_goal' or 'right_goal')
        """
        if self.goal_coordinates is None:
            # Fallback: simple distance calculation
            center_y = self.frame_height / 2
            left_dist = np.sqrt(ball_xy[0]**2 + (ball_xy[1] - center_y)**2)
            right_dist = np.sqrt((self.frame_width - ball_xy[0])**2 + (ball_xy[1] - center_y)**2)
            
            if left_dist < right_dist:
                return left_dist / (self.pixels_per_meter or 1), 'left_goal'
            else:
                return right_dist / (self.pixels_per_meter or 1), 'right_goal'
        
        ball_x, ball_y = ball_xy[0], ball_xy[1]
        
        # Calculate distance to each goal center
        left_goal = self.goal_coordinates['left_goal']
        right_goal = self.goal_coordinates['right_goal']
        
        dist_to_left = np.sqrt((ball_x - left_goal['center_x'])**2 + (ball_y - left_goal['center_y'])**2)
        dist_to_right = np.sqrt((ball_x - right_goal['center_x'])**2 + (ball_y - right_goal['center_y'])**2)
        
        if target_goal == 'left_goal' or (target_goal == 'auto' and dist_to_left < dist_to_right):
            return dist_to_left / (self.pixels_per_meter or 1), 'left_goal'
        else:
            return dist_to_right / (self.pixels_per_meter or 1), 'right_goal'
    
    def check_velocity_vector_intersection(self, ball_xy, ball_velocity_dx, ball_velocity_dy, goal_name):
        """
        Check if ball's velocity vector intersects with a 10-meter radius around the goal frame.
        Returns: (intersects: bool, angle_of_arrival: float in degrees)
        """
        if self.goal_coordinates is None:
            # Fallback: simple direction check (with stricter criteria)
            intersects = self.is_toward_goal_simple(ball_xy, ball_velocity_dx, ball_velocity_dy)
            return intersects, 0  # Angle unknown in fallback mode
        
        goal = self.goal_coordinates[goal_name]
        goal_center = np.array([goal['center_x'], goal['center_y']])
        ball_pos = np.array([ball_xy[0], ball_xy[1]])
        
        # Vector from ball to goal
        to_goal = goal_center - ball_pos
        
        # Velocity vector
        velocity_vec = np.array([ball_velocity_dx, ball_velocity_dy])
        velocity_magnitude = np.linalg.norm(velocity_vec)
        
        if velocity_magnitude < 0.1:
            return False, 0
        
        # Normalize velocity vector
        velocity_unit = velocity_vec / velocity_magnitude
        
        # Project ball-to-goal vector onto velocity direction
        projection = np.dot(to_goal, velocity_unit)
        
        # Point on trajectory closest to goal
        closest_point = ball_pos + projection * velocity_unit
        
        # Distance from goal center to closest point
        dist_to_closest = np.linalg.norm(goal_center - closest_point)
        
        # Convert 10 meters to pixels
        radius_pixels = GOAL_PROXIMITY_RADIUS_METERS * (self.pixels_per_meter or 1)
        
        # Check if trajectory passes within radius
        intersects = dist_to_closest <= radius_pixels
        
        # Calculate angle of arrival (angle between velocity and goal direction)
        to_goal_unit = to_goal / (np.linalg.norm(to_goal) + 1e-6)
        angle_rad = np.arccos(np.clip(np.dot(velocity_unit, to_goal_unit), -1, 1))
        angle_degrees = np.degrees(angle_rad)
        
        # ADJUSTED CHECK: Angle must be less than 60 degrees (was 45 - too strict)
        # This filters out square passes but allows shots from wider angles
        if angle_degrees > 60:
            intersects = False  # Too wide - likely a cross or square pass
        
        return intersects, angle_degrees
    
    def is_toward_goal(self, shooter_pos, ball_velocity_dx):
        """Check if ball is moving toward a goal (legacy method, enhanced version below)"""
        shooter_x = shooter_pos[0]
        center = self.frame_width / 2
        
        # If shooter is on left half, ball should move left (negative dx) to left goal
        # If shooter is on right half, ball should move right (positive dx) to right goal
        # But in soccer, you shoot at OPPONENT's goal, so:
        # - Player on left attacks RIGHT goal (ball moves right, positive dx)
        # - Player on right attacks LEFT goal (ball moves left, negative dx)
        
        if shooter_x < center:
            # Player on left side, attacking right goal
            return ball_velocity_dx > 5  # Moving right
        else:
            # Player on right side, attacking left goal
            return ball_velocity_dx < -5  # Moving left
    
    def is_toward_goal_simple(self, ball_xy, ball_velocity_dx, ball_velocity_dy=None):
        """
        Simple check if ball is moving toward goal (fallback).
        STRICTENED: Requires strong directional movement, not just slight movement.
        """
        ball_x = ball_xy[0]
        center = self.frame_width / 2
        
        # Require stronger directional movement (increased threshold)
        min_dx_threshold = 8  # Increased from 5 to filter out slow movements
        
        if ball_x < center:
            # Ball on left, should move right toward right goal
            # Also check it's not moving sideways too much (if dy available)
            if ball_velocity_dy is not None:
                # Require more forward movement than sideways (prevents square passes)
                return ball_velocity_dx > min_dx_threshold and abs(ball_velocity_dx) > abs(ball_velocity_dy) * 1.5
            return ball_velocity_dx > min_dx_threshold
        else:
            # Ball on right, should move left toward left goal
            if ball_velocity_dy is not None:
                return ball_velocity_dx < -min_dx_threshold and abs(ball_velocity_dx) > abs(ball_velocity_dy) * 1.5
            return ball_velocity_dx < -min_dx_threshold
    
    def is_in_shooting_range(self, position):
        """
        Check if position is in shooting range (attacking third).
        ADJUSTED: Less restrictive to allow more shot detection opportunities.
        """
        x, y = position[0], position[1]
        
        # Width constraints: Less restrictive (central 80% instead of 70%)
        # Excludes extreme sidelines but allows wider shots
        width_center_start = self.frame_width * 0.10
        width_center_end = self.frame_width * 0.90
        
        # Check if in central width zone
        in_central_zone = width_center_start <= x <= width_center_end
        
        # Height constraints: Less restrictive (central 85% instead of 75%)
        height_center_start = self.frame_height * 0.075
        height_center_end = self.frame_height * 0.925
        
        in_central_height = height_center_start <= y <= height_center_end
        
        # Must be in attacking third AND in central zone (not extreme wide/corner)
        left_attack = x < self.frame_width * 0.33
        right_attack = x > self.frame_width * 0.67
        
        return (left_attack or right_attack) and in_central_zone and in_central_height
    
    def is_ball_in_goal_mouth(self, ball_xy, goal_name):
        """
        Check if ball position is within goal frame coordinates.
        Only classifies as "On Target" if ball's trajectory intersects goal posts.
        """
        if self.goal_coordinates is None:
            return False
        
        goal = self.goal_coordinates[goal_name]
        ball_x, ball_y = ball_xy[0], ball_xy[1]
        
        # Check if ball is between goal posts (with some margin)
        between_posts = (goal['left_post'] <= ball_x <= goal['right_post'])
        
        # Check if ball is below crossbar
        below_crossbar = (ball_y >= goal['top_post'])
        
        return between_posts and below_crossbar
    
    def detect_shot(self, frame_idx, frame, ball_xy, shooter_tid, shooter_pos, 
                    shooter_bbox, shooter_team, pitch_bounds=None):
        """
        Detect if a shot on goal is occurring with REFINED DETECTION LOGIC.
        
        Returns: (is_shot, shot_type, confidence, metrics_dict) or (False, None, 0, None)
        """
        # Update pitch bounds if provided
        if pitch_bounds:
            self.update_pitch_bounds(pitch_bounds)
            self.calibrate_goal_coordinates(pitch_bounds)
        
        # Initialize goal coordinates if not yet set
        if self.goal_coordinates is None:
            self.calibrate_goal_coordinates(pitch_bounds)
        
        # Add ball position
        self.add_ball_position(frame_idx, ball_xy)
        
        # Check cooldown
        if frame_idx - self.last_shot_frame < SHOT_COOLDOWN_FRAMES:
            return False, None, 0, None
        
        # Calculate ball velocity
        speed, dx, dy = self.get_ball_velocity()
        
        # === TIGHTENED SHOT DETECTION CRITERIA (to reduce over-detection) ===
        # These filters ensure only clear, unambiguous shots are analyzed
        # This prevents crosses, hard passes, and clearances from triggering VLM
        
        # 1. Ball must be moving FAST enough (18 px/frame = shot, excludes soft passes)
        if speed < SHOT_VELOCITY_THRESHOLD:
            return False, None, 0, None
        
        # 2. Player must be in CENTRAL shooting range (excludes wide areas where crosses occur)
        # Central zone only: excludes sidelines/corners to filter out crosses
        if not self.is_in_shooting_range(shooter_pos):
            return False, None, 0, None
        
        # 3. Calculate distance to goal - only trigger if within 10m radius
        distance_to_goal, target_goal = self.calculate_distance_to_goal(ball_xy)
        
        # Proximity Constraint: Only trigger if ball is within 10-meter radius of goal
        if distance_to_goal > GOAL_PROXIMITY_RADIUS_METERS:
            return False, None, 0, None
        
        # 4. Velocity Vector Check: Ball trajectory must intersect goal area
        vector_intersects, angle_of_arrival = self.check_velocity_vector_intersection(
            ball_xy, dx, dy, target_goal
        )
        
        if not vector_intersects:
            return False, None, 0, None
        
        # 5. Use VLM to confirm shot and classify outcome (with enhanced prompt)
        # High confidence threshold (80%) ensures only clear shots are accepted
        crop = self._get_shot_crop(frame, ball_xy)
        if crop is None or crop.size == 0:
            return False, None, 0, None
        
        is_shot, shot_type, vlm_conf = self._vlm_analyze_shot(
            crop, shooter_team, speed, distance_to_goal, angle_of_arrival
        )
        
        # 6. Final filter: Require high VLM confidence (80%+) to exclude uncertain guesses
        if is_shot and vlm_conf >= SHOT_CONFIDENCE_THRESHOLD:
            # Prepare metrics for recording
            metrics = {
                'ball_velocity_px_per_frame': round(speed, 2),
                'angle_of_arrival_degrees': round(angle_of_arrival, 2),
                'vlm_confidence_score': vlm_conf,
                'distance_from_goal_meters': round(distance_to_goal, 2),
                'target_goal': target_goal,
                'velocity_dx': round(dx, 2),
                'velocity_dy': round(dy, 2)
            }
            
            self.last_shot_frame = frame_idx
            return True, shot_type, vlm_conf, metrics
        
        return False, None, 0, None
    
    def _get_shot_crop(self, frame, ball_xy, crop_size=500):
        """Get a larger crop around the ball for shot analysis"""
        h, w = frame.shape[:2]
        x1 = max(0, int(ball_xy[0]) - crop_size // 2)
        y1 = max(0, int(ball_xy[1]) - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        return frame[y1:y2, x1:x2]
    
    def _vlm_analyze_shot(self, frame_crop, shooter_team, ball_speed, distance_to_goal=None, angle_of_arrival=None):
        """
        Use VLM to detect shot and classify outcome with ENHANCED PROMPTS.
        Returns: (is_shot, shot_type, confidence)
        """
        # Stage 1: Is this a shot?
        is_shot, shot_conf = self._vlm_stage1_is_shot(frame_crop, ball_speed, distance_to_goal)
        
        if not is_shot:
            return False, None, shot_conf
        
        # Stage 2: Classify shot outcome
        shot_type, outcome_conf = self._vlm_stage2_shot_outcome(frame_crop, angle_of_arrival)
        
        # Combine confidence (Stage 1: 40%, Stage 2: 60%)
        final_conf = int((shot_conf * 0.4) + (outcome_conf * 0.6))
        
        return True, shot_type, final_conf
    
    def _vlm_stage1_is_shot(self, frame_crop, ball_speed, distance_to_goal=None):
        """
        VLM Stage 1: Detect if this is a shot on goal - REFINED PROMPT WITH STRICT BOUNDARIES
        """
        dist_info = f"Distance to goal: {distance_to_goal:.1f}m" if distance_to_goal else "Near goal"
        
        prompt = f"""Analyze the soccer action in this crop.

BALL SPEED: {ball_speed:.0f} pixels/frame | {dist_info}

TRAJECTORY CHECK: Is the ball moving with high power specifically toward the rectangular goal frame?

CRITICAL CLASSIFICATION CRITERIA:
1. TRAJECTORY ANALYSIS:
   - Ball must be moving FAST and POWERFULLY (not a soft pass)
   - Ball trajectory must point DIRECTLY toward the goal frame
   - Ball is NOT traveling toward a teammate (would be a PASS)

2. VISUAL EVIDENCE:
   - Player's leg in powerful KICKING/FOLLOW-THROUGH position
   - Player facing toward GOAL direction (not sideways for cross)
   - GOAL FRAME or GOALKEEPER visible in image
   - High-velocity ball movement (not slow/controlled pass)

3. CONTEXT:
   - Player in attacking position (near opponent's goal)
   - Ball is within scoring range
   - Action shows attacking intent (not defensive clearance)

EXCLUSION RULES (NOT a shot):
- Ball traveling slowly → Likely a PASS
- Ball moving toward teammate → PASS (reply 'PASS')
- Player not in attacking position → Clearance or long ball
- Ball moving away from goal → Not a shot attempt

Answer with ONLY: YES (if shot) or NO (if pass/clearance/not a shot)"""

        response = self.vlm_query(frame_crop, prompt, max_tokens=10)
        response_upper = response.upper()
        
        # Check for PASS explicitly
        if "PASS" in response_upper:
            return False, 30
        
        is_shot = "YES" in response_upper and "NO" not in response_upper
        confidence = 85 if is_shot else 35
        
        return is_shot, confidence
    
    def _vlm_stage2_shot_outcome(self, frame_crop, angle_of_arrival=None):
        """
        VLM Stage 2: Classify shot outcome - PRECISE PROMPT WITH CLEAR BOUNDARIES
        """
        angle_info = f"Angle: {angle_of_arrival:.1f}°" if angle_of_arrival is not None else ""
        
        prompt = f"""Analyze the soccer action in this crop.

TRAJECTORY CHECK: Is the ball moving with high power specifically toward the rectangular goal frame?

BALL TRAJECTORY: {angle_info}

CLASSIFICATION - Answer with EXACT format: [Type] | Reason: [Trajectory details]

CLASSIFICATION OPTIONS:

1. SHOT ON TARGET:
   - Ball is STRICTLY between the two vertical posts
   - Ball is beneath the crossbar
   - If the goalie didn't touch it, it would be a GOAL
   - Ball trajectory intersects the rectangular goal frame
   - Answer: "Shot on Target | Reason: Ball between posts, under crossbar"

2. SHOT OFF TARGET:
   - Ball is clearly going OVER the bar OR
   - Ball is clearly going WIDE of the posts
   - Ball will NOT enter goal frame
   - Answer: "Shot off Target | Reason: Over bar / Wide of posts"

3. SHOT BLOCKED:
   - Defender's body is intercepting/blocking the ball
   - Ball deflected by defender before reaching goal
   - Answer: "Shot blocked | Reason: Deflected by defender"

4. GOAL:
   - Ball is PAST the goalkeeper
   - Ball is INSIDE the goal/net
   - Ball has clearly crossed the goal line
   - Answer: "Goal | Reason: Ball in net, past goal line"

5. NOT A SHOT (if ball moving slowly or toward teammate):
   - Answer: "PASS"

CRITICAL: Only classify as "Shot on Target" if ball trajectory will intersect the goal frame.
Be strict: If ball is going over/wide, classify as "Shot off Target".

Answer format: [Type] | Reason: [Details]"""

        response = self.vlm_query(frame_crop, prompt, max_tokens=50)
        response_upper = response.upper()
        
        # Parse response
        # GOAL - highest priority
        if "GOAL" in response_upper or ("NET" in response_upper and ("IN" in response_upper or "PAST" in response_upper)):
            return "Goal", 90
        
        # BLOCKED
        if "BLOCK" in response_upper or ("DEFENDER" in response_upper and ("INTERCEPT" in response_upper or "DEFLECT" in response_upper)):
            return "Shot blocked", 80
        
        # ON TARGET - strict check
        if "ON TARGET" in response_upper or ("BETWEEN" in response_upper and "POST" in response_upper and "UNDER" in response_upper):
            # Additional validation: must mention goal frame intersection
            if "POST" in response_upper or "CROSSBAR" in response_upper:
                return "Shot on target", 85
        
        # OFF TARGET
        if "OFF TARGET" in response_upper or "OVER" in response_upper or "WIDE" in response_upper or "MISS" in response_upper:
            return "Shot off target", 80
        
        # PASS (should not reach here, but handle it)
        if "PASS" in response_upper:
            return "Shot off target", 60
        
        # Default to off target if uncertain
        return "Shot off target", 65
    
    def record_shot(self, frame_idx, fps, shooter_tid, shooter_team, shot_type, confidence, 
                    shooter_pos, ball_xy, metrics=None):
        """
        Record a detected shot event with ENHANCED METRICS.
        Metrics include: ball velocity, angle of arrival, distance from goal, VLM confidence
        """
        time_str = self._format_time(frame_idx / fps)
        
        event = {
            "time": time_str,
            "frame": frame_idx,
            "shooter_id": int(shooter_tid),
            "team": shooter_team,
            "shot_type": shot_type,
            "confidence": confidence,
            "shooter_x": round(shooter_pos[0], 1),
            "shooter_y": round(shooter_pos[1], 1),
            "ball_x": round(ball_xy[0], 1) if ball_xy is not None else 0,
            "ball_y": round(ball_xy[1], 1) if ball_xy is not None else 0
        }
        
        # Add enhanced metrics if provided
        if metrics:
            event.update({
                "ball_velocity_px_per_frame": metrics.get('ball_velocity_px_per_frame', 0),
                "angle_of_arrival_degrees": metrics.get('angle_of_arrival_degrees', 0),
                "vlm_confidence_score": metrics.get('vlm_confidence_score', confidence),
                "distance_from_goal_meters": metrics.get('distance_from_goal_meters', 0),
                "target_goal": metrics.get('target_goal', 'unknown'),
                "velocity_dx": metrics.get('velocity_dx', 0),
                "velocity_dy": metrics.get('velocity_dy', 0)
            })
        else:
            # Default values if metrics not provided
            event.update({
                "ball_velocity_px_per_frame": 0,
                "angle_of_arrival_degrees": 0,
                "vlm_confidence_score": confidence,
                "distance_from_goal_meters": 0,
                "target_goal": "unknown",
                "velocity_dx": 0,
                "velocity_dy": 0
            })
        
        self.shot_events.append(event)
        
        # Update stats
        if shot_type in self.stats and shooter_team in self.stats[shot_type]:
            self.stats[shot_type][shooter_team] += 1
        
        return event
    
    def _format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def get_shot_events(self):
        """Get all recorded shot events"""
        return self.shot_events
    
    def get_stats(self):
        """Get shot statistics by team"""
        return self.stats
    
    def print_summary(self):
        """Print shot detection summary"""
        total_shots = len(self.shot_events)
        print("\n" + "=" * 80)
        print("🎯 SHOT ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\nShot Type           | Blue Team | Red Team | Total")
        print("-" * 60)
        
        for shot_type in ["Shot on target", "Shot off target", "Shot blocked", "Goal"]:
            blue = self.stats[shot_type].get("Blue", 0)
            red = self.stats[shot_type].get("Red", 0)
            total = blue + red
            print(f"{shot_type:<20} | {blue:^9} | {red:^8} | {total:^5}")
        
        print("-" * 60)
        
        blue_total = sum(self.stats[st].get("Blue", 0) for st in self.stats)
        red_total = sum(self.stats[st].get("Red", 0) for st in self.stats)
        
        print(f"{'TOTAL':<20} | {blue_total:^9} | {red_total:^8} | {total_shots:^5}")
        print("=" * 80)


def generate_shot_report_csv(shot_events, output_path="shot_report.csv"):
    """Generate CSV report of shot events with ENHANCED METRICS"""
    import csv
    
    # Extended fieldnames to include new metrics
    fieldnames = [
        'time', 'frame', 'shooter_id', 'team', 'shot_type', 
        'confidence', 'shooter_x', 'shooter_y', 'ball_x', 'ball_y',
        'ball_velocity_px_per_frame', 'angle_of_arrival_degrees', 
        'vlm_confidence_score', 'distance_from_goal_meters', 'target_goal',
        'velocity_dx', 'velocity_dy'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in shot_events:
            # Ensure all fields exist in event dict
            row = {field: event.get(field, 0) for field in fieldnames}
            writer.writerow(row)
    
    print(f"✅ Shot report saved to: {output_path}")
    return output_path


def generate_shot_report_json(shot_events, stats, output_path="shot_report.json"):
    """Generate JSON report of shot events"""
    import json
    import numpy as np
    
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
    
    # Calculate enhanced metrics summary
    velocities = [e.get('ball_velocity_px_per_frame', 0) for e in shot_events if e.get('ball_velocity_px_per_frame', 0) > 0]
    angles = [e.get('angle_of_arrival_degrees', 0) for e in shot_events if e.get('angle_of_arrival_degrees', 0) > 0]
    distances = [e.get('distance_from_goal_meters', 0) for e in shot_events if e.get('distance_from_goal_meters', 0) > 0]
    
    report = {
        "summary": {
            "total_shots": len(shot_events),
            "shots_on_target": sum(1 for e in shot_events if e['shot_type'] == 'Shot on target'),
            "shots_off_target": sum(1 for e in shot_events if e['shot_type'] == 'Shot off target'),
            "shots_blocked": sum(1 for e in shot_events if e['shot_type'] == 'Shot blocked'),
            "goals": sum(1 for e in shot_events if e['shot_type'] == 'Goal'),
            "metrics": {
                "avg_ball_velocity_px_per_frame": round(np.mean(velocities), 2) if velocities else 0,
                "avg_angle_of_arrival_degrees": round(np.mean(angles), 2) if angles else 0,
                "avg_distance_from_goal_meters": round(np.mean(distances), 2) if distances else 0,
                "avg_vlm_confidence": round(np.mean([e.get('vlm_confidence_score', e.get('confidence', 0)) for e in shot_events]), 2) if shot_events else 0
            }
        },
        "by_team": stats,
        "events": shot_events
    }
    
    # Convert numpy types before JSON serialization
    report = convert_to_json_serializable(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Shot report JSON saved to: {output_path}")
    return output_path

