"""
ScoutMe - Shot on Goal Detection Module
Detects shots on goal and classifies them as ON TARGET or OFF TARGET
Uses VLM (Molmo-7B) for accurate visual analysis

Integration: Import and use with main5.py
"""

import numpy as np
from collections import deque


# === SHOT DETECTION PARAMETERS ===
SHOT_PROXIMITY_THRESHOLD = 100         # Ball near player for potential shot (increased)
SHOT_VELOCITY_THRESHOLD = 15           # Minimum ball speed for shot (LOWERED from 25)
GOAL_ZONE_MARGIN = 0.25                # 25% from each end = goal areas (increased)
SHOT_COOLDOWN_FRAMES = 45              # 1.5 seconds between shots (reduced)
MIN_SHOT_DISTANCE_TO_GOAL = 50         # Minimum distance for shot (reduced)
SHOT_CONFIDENCE_THRESHOLD = 60         # Minimum confidence (lowered)


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
        else:
            # Use frame-based detection
            margin = self.frame_width * GOAL_ZONE_MARGIN
            self.left_goal_zone = (0, margin)
            self.right_goal_zone = (self.frame_width - margin, self.frame_width)
    
    def update_pitch_bounds(self, pitch_bounds):
        """Update goal zones when pitch bounds are detected"""
        if pitch_bounds and pitch_bounds.get('detected', False):
            self._setup_goal_zones(pitch_bounds)
    
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
    
    def is_toward_goal(self, shooter_pos, ball_velocity_dx):
        """Check if ball is moving toward a goal"""
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
    
    def is_in_shooting_range(self, position):
        """Check if position is in shooting range (attacking third)"""
        x = position[0]
        # Attacking third = last 33% of the pitch toward goal
        left_attack = x < self.frame_width * 0.33
        right_attack = x > self.frame_width * 0.67
        return left_attack or right_attack
    
    def detect_shot(self, frame_idx, frame, ball_xy, shooter_tid, shooter_pos, 
                    shooter_bbox, shooter_team, pitch_bounds=None):
        """
        Detect if a shot on goal is occurring.
        
        Returns: (is_shot, shot_type, confidence) or (False, None, 0)
        """
        # Update pitch bounds if provided
        if pitch_bounds:
            self.update_pitch_bounds(pitch_bounds)
        
        # Add ball position
        self.add_ball_position(frame_idx, ball_xy)
        
        # Check cooldown
        if frame_idx - self.last_shot_frame < SHOT_COOLDOWN_FRAMES:
            return False, None, 0
        
        # Calculate ball velocity
        speed, dx, dy = self.get_ball_velocity()
        
        # === SHOT DETECTION CRITERIA ===
        
        # 1. Ball must be moving fast enough
        if speed < SHOT_VELOCITY_THRESHOLD:
            return False, None, 0
        
        # 2. Player must be in shooting range
        if not self.is_in_shooting_range(shooter_pos):
            return False, None, 0
        
        # 3. Ball must be moving toward goal
        if not self.is_toward_goal(shooter_pos, dx):
            return False, None, 0
        
        # 4. Use VLM to confirm shot and classify outcome
        crop = self._get_shot_crop(frame, ball_xy)
        if crop is None or crop.size == 0:
            return False, None, 0
        
        is_shot, shot_type, vlm_conf = self._vlm_analyze_shot(crop, shooter_team, speed)
        
        if is_shot and vlm_conf >= SHOT_CONFIDENCE_THRESHOLD:
            self.last_shot_frame = frame_idx
            return True, shot_type, vlm_conf
        
        return False, None, 0
    
    def _get_shot_crop(self, frame, ball_xy, crop_size=500):
        """Get a larger crop around the ball for shot analysis"""
        h, w = frame.shape[:2]
        x1 = max(0, int(ball_xy[0]) - crop_size // 2)
        y1 = max(0, int(ball_xy[1]) - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        return frame[y1:y2, x1:x2]
    
    def _vlm_analyze_shot(self, frame_crop, shooter_team, ball_speed):
        """
        Use VLM to detect shot and classify outcome.
        Returns: (is_shot, shot_type, confidence)
        """
        # Stage 1: Is this a shot?
        is_shot, shot_conf = self._vlm_stage1_is_shot(frame_crop, ball_speed)
        
        if not is_shot:
            return False, None, shot_conf
        
        # Stage 2: Classify shot outcome
        shot_type, outcome_conf = self._vlm_stage2_shot_outcome(frame_crop)
        
        # Combine confidence
        final_conf = int((shot_conf * 0.4) + (outcome_conf * 0.6))
        
        return True, shot_type, final_conf
    
    def _vlm_stage1_is_shot(self, frame_crop, ball_speed):
        """
        VLM Stage 1: Detect if this is a shot on goal - ENHANCED PROMPT
        """
        prompt = f"""SOCCER SHOT DETECTION - Analyze this image very carefully.

BALL SPEED: {ball_speed:.0f} pixels/frame (HIGH SPEED indicates potential shot)

LOOK AT THE IMAGE TO DETERMINE: Is this a SHOT ON GOAL?

STEP 1: CHECK PLAYER'S ACTION:
   - Is the player KICKING the ball with POWER/HARD force?
   - Is the player's leg in FOLLOW-THROUGH position (extended leg after kick)?
   - Is the player facing toward the GOAL direction?
   
STEP 2: CHECK BALL TRAJECTORY:
   - Is the ball moving FAST toward the GOAL?
   - Is the ball traveling toward the GOAL POSTS or GOAL AREA?
   - Is the ball NOT traveling toward a teammate (not a pass)?

STEP 3: CHECK FIELD POSITION:
   - Is the player in SHOOTING POSITION (near goal, in attacking area)?
   - Is the GOAL FRAME visible in the image?
   - Is a GOALKEEPER visible (indicating shot toward goal)?

SHOT vs PASS DIFFERENCE:
   - SHOT = Player kicks ball HARD/POWERFULLY toward GOAL, attacking intent
   - PASS = Player passes ball to TEAMMATE, ball travels toward another player

VISUAL CUES FOR SHOT:
   - Powerful kicking motion (leg fully extended, strong follow-through)
   - Ball trajectory heading toward GOAL FRAME
   - Player in attacking/forward position
   - Goal or goalkeeper visible in frame
   - High ball speed
   - Ball NOT going to nearby teammate

Answer: Is this a SHOT ON GOAL? Reply only YES or NO."""

        response = self.vlm_query(frame_crop, prompt, max_tokens=10)
        response_upper = response.upper()
        
        is_shot = "YES" in response_upper
        confidence = 80 if is_shot else 40
        
        return is_shot, confidence
    
    def _vlm_stage2_shot_outcome(self, frame_crop):
        """
        VLM Stage 2: Classify shot outcome - ENHANCED PROMPT FOR 100% UNDERSTANDING
        """
        prompt = """SHOT OUTCOME ANALYSIS - Look at this soccer shot image carefully.

A SHOT ON GOAL has been detected. Now classify WHERE the ball is going:

STEP 1: LOOK AT THE GOAL FRAME:
   - Identify the GOAL POSTS (left post, right post)
   - Identify the CROSSBAR (top horizontal bar)
   - Identify the GOAL NET (if visible)

STEP 2: TRACE THE BALL TRAJECTORY:
   - Where is the ball NOW in the image?
   - Where is the ball HEADING toward?
   - Will the ball go INTO the goal or MISS the goal?

STEP 3: CHECK FOR DEFENDERS:
   - Is a DEFENDER blocking/intercepting the ball?
   - Is the ball being DEFLECTED by a defender's body?

STEP 4: CHECK GOALKEEPER POSITION:
   - Is the ball PAST the goalkeeper?
   - Is the goalkeeper BEATEN (ball already past)?

CLASSIFY THE SHOT OUTCOME:

(A) SHOT ON TARGET - Ball heading INTO the goal frame
    VISUAL CUES:
    - Ball trajectory is heading BETWEEN the two goal posts
    - Ball is heading UNDER the crossbar (not over)
    - Ball would go INTO the goal if not saved
    - Goalkeeper would need to make a SAVE
    - Ball is THREATENING the goal
    
(B) SHOT OFF TARGET - Ball MISSING the goal frame
    VISUAL CUES:
    - Ball is going WIDE (to the left or right of posts)
    - Ball is going OVER the crossbar (too high)
    - Ball will NOT enter the goal frame
    - Ball will NOT threaten the goalkeeper
    - Ball is clearly MISSING the goal
    
(C) SHOT BLOCKED - Ball is being BLOCKED by a defender
    VISUAL CUES:
    - A DEFENDER's body is intercepting the ball
    - Ball is being DEFLECTED or STOPPED by defender
    - Defender's leg/body is in the way of the ball
    - Ball is NOT reaching the goal due to defender
    
(D) GOAL - Ball is GOING INTO THE NET
    VISUAL CUES:
    - Ball is PAST the goalkeeper (goalkeeper beaten)
    - Ball is CLEARLY entering the goal/net
    - Ball is INSIDE the goal frame, past the goal line
    - Goal NET is visible and ball is in/entering it
    - This is a SCORED GOAL

LOOK AT THE IMAGE:
1. Check ball position relative to goal posts
2. Check if ball is going into goal or missing
3. Check if any defender is blocking
4. Check if ball is past goalkeeper

ANSWER WITH ONLY ONE LETTER (A, B, C, or D):
My classification:"""

        response = self.vlm_query(frame_crop, prompt, max_tokens=20)
        response_upper = response.upper().strip()
        
        # Extract classification - check for letter first
        first_char = ''
        for c in response_upper:
            if c in 'ABCD':
                first_char = c
                break
        
        # Enhanced parsing with keyword fallback
        # GOAL (D) - highest priority
        if first_char == 'D' or 'GOAL' in response_upper or ('NET' in response_upper and 'ENTER' in response_upper):
            return "Goal", 90
        
        # BLOCKED (C) - check for defender blocking
        if first_char == 'C' or 'BLOCK' in response_upper or ('DEFENDER' in response_upper and 'INTERCEPT' in response_upper):
            return "Shot blocked", 80
        
        # ON TARGET (A) - ball heading into goal frame
        if first_char == 'A' or ('ON' in response_upper and 'TARGET' in response_upper) or ('BETWEEN' in response_upper and 'POST' in response_upper):
            return "Shot on target", 85
        
        # OFF TARGET (B) - ball missing goal
        if first_char == 'B' or 'OFF' in response_upper or 'MISS' in response_upper or 'WIDE' in response_upper or 'OVER' in response_upper:
            return "Shot off target", 80
        
        # Default to off target if uncertain
        return "Shot off target", 65
    
    def record_shot(self, frame_idx, fps, shooter_tid, shooter_team, shot_type, confidence, 
                    shooter_pos, ball_xy):
        """Record a detected shot event"""
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
    """Generate CSV report of shot events"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time', 'frame', 'shooter_id', 'team', 'shot_type', 
            'confidence', 'shooter_x', 'shooter_y', 'ball_x', 'ball_y'
        ])
        writer.writeheader()
        for event in shot_events:
            writer.writerow(event)
    
    print(f"✅ Shot report saved to: {output_path}")
    return output_path


def generate_shot_report_json(shot_events, stats, output_path="shot_report.json"):
    """Generate JSON report of shot events"""
    import json
    
    report = {
        "summary": {
            "total_shots": len(shot_events),
            "shots_on_target": sum(1 for e in shot_events if e['shot_type'] == 'Shot on target'),
            "shots_off_target": sum(1 for e in shot_events if e['shot_type'] == 'Shot off target'),
            "shots_blocked": sum(1 for e in shot_events if e['shot_type'] == 'Shot blocked'),
            "goals": sum(1 for e in shot_events if e['shot_type'] == 'Goal')
        },
        "by_team": stats,
        "events": shot_events
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Shot report JSON saved to: {output_path}")
    return output_path

