"""
ScoutMe - Half Detection Module
Automatically detects match halves (1st half start/end, 2nd half start/end)
Uses activity analysis, kickoff detection, and VLM verification

CEO REQUIREMENT:
- stamp time of first half start & end time
- stamp time of second half start & end time

UPDATED: Now supports pre-computed boundaries from MatchSegmenter
for faster processing (skips per-frame detection when boundaries are known)
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, List
import cv2


class HalfDetector:
    """
    Detects match half boundaries by analyzing:
    1. Player activity levels (playing vs break)
    2. Kickoff detection (center circle + formation)
    3. VLM verification for key moments
    
    OR uses pre-computed boundaries from MatchSegmenter (faster).
    """
    
    def __init__(self, frame_width: int, frame_height: int, fps: float, vlm_query_func=None,
                 precomputed_boundaries: Optional[Dict] = None):
        """
        Initialize Half Detector.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Video frames per second
            vlm_query_func: Optional VLM function for verification
            precomputed_boundaries: Optional dict from MatchSegmenter with pre-detected boundaries
                                   {"h1_start_frame": N, "h1_end_frame": M, "h2_start_frame": X, "h2_end_frame": Y}
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.vlm_query = vlm_query_func
        
        # Half boundaries (frame numbers)
        self.h1_start: Optional[int] = None
        self.h1_end: Optional[int] = None
        self.h2_start: Optional[int] = None
        self.h2_end: Optional[int] = None
        
        # Flag to indicate if boundaries were pre-computed
        self.using_precomputed = False
        
        # Apply pre-computed boundaries if provided
        if precomputed_boundaries:
            self._apply_precomputed_boundaries(precomputed_boundaries)
        
        # Detection state
        self.activity_history = deque(maxlen=300)  # ~10-12 seconds of activity
        self.current_state = "unknown"  # "playing", "break", "halftime", "warmup"
        self.state_start_frame = 0
        
        # Thresholds (from config)
        self.playing_threshold = 0.65  # Activity above this = active play
        self.break_threshold = 0.30    # Activity below this = break
        self.min_half_duration_frames = 40 * 60 * fps  # 40 minutes minimum
        self.max_half_duration_frames = 50 * 60 * fps  # 50 minutes maximum
        self.min_state_duration_frames = 15 * fps      # 15 seconds to confirm state
        self.min_break_duration_frames = 5 * 60 * fps  # 5 minutes minimum halftime
        
        # Center circle detection
        self.center_x = frame_width / 2
        self.center_y = frame_height / 2
        self.center_circle_radius = 100  # pixels
        
        # Kickoff candidates (frame, confidence)
        self.kickoff_candidates = []
        
        # Stats
        self.total_activity_samples = 0
        self.kickoff_detected_count = 0
    
    def _apply_precomputed_boundaries(self, boundaries: Dict):
        """Apply pre-computed boundaries from MatchSegmenter."""
        self.h1_start = boundaries.get('h1_start_frame')
        self.h1_end = boundaries.get('h1_end_frame')
        self.h2_start = boundaries.get('h2_start_frame')
        self.h2_end = boundaries.get('h2_end_frame')
        self.using_precomputed = True
        
        print(f"📋 HalfDetector: Using pre-computed boundaries")
        if self.h1_start is not None:
            print(f"   H1: Frame {self.h1_start} - {self.h1_end}")
        if self.h2_start is not None:
            print(f"   H2: Frame {self.h2_start} - {self.h2_end}")
    
    def set_boundaries(self, h1_start: int = None, h1_end: int = None,
                      h2_start: int = None, h2_end: int = None):
        """
        Manually set half boundaries (e.g., from MatchSegmenter).
        Skips per-frame detection when boundaries are set.
        """
        if h1_start is not None:
            self.h1_start = h1_start
        if h1_end is not None:
            self.h1_end = h1_end
        if h2_start is not None:
            self.h2_start = h2_start
        if h2_end is not None:
            self.h2_end = h2_end
        
        self.using_precomputed = True
    
    def calculate_activity_score(self, player_positions: List[Tuple[float, float]], 
                                  ball_xy: Optional[Tuple[float, float]],
                                  prev_positions: Optional[Dict] = None) -> float:
        """
        Calculate activity score based on player movement and ball presence.
        Returns 0.0 (no activity) to 1.0 (high activity).
        """
        if not player_positions:
            return 0.0
        
        score = 0.0
        factors = []
        
        # Factor 1: Number of players on pitch (expect 20-22)
        num_players = len(player_positions)
        if num_players >= 18:
            factors.append(1.0)  # Full team present
        elif num_players >= 10:
            factors.append(0.7)  # Partial team
        else:
            factors.append(0.3)  # Too few players (break/warmup)
        
        # Factor 2: Player spread (high spread = active positioning)
        if num_players >= 4:
            x_coords = [p[0] for p in player_positions]
            y_coords = [p[1] for p in player_positions]
            x_spread = (max(x_coords) - min(x_coords)) / self.frame_width
            y_spread = (max(y_coords) - min(y_coords)) / self.frame_height
            spread_score = min(1.0, (x_spread + y_spread) / 1.5)
            factors.append(spread_score)
        
        # Factor 3: Ball presence
        if ball_xy is not None:
            factors.append(1.0)
        else:
            factors.append(0.4)  # Ball not detected
        
        # Factor 4: Player movement (if previous positions available)
        if prev_positions and len(prev_positions) > 0:
            total_movement = 0
            movement_count = 0
            for pos in player_positions:
                # Find closest previous position
                for prev_pos in prev_positions.values():
                    if 'position' in prev_pos:
                        dist = np.linalg.norm(np.array(pos) - np.array(prev_pos['position']))
                        total_movement += dist
                        movement_count += 1
                        break
            
            if movement_count > 0:
                avg_movement = total_movement / movement_count
                movement_score = min(1.0, avg_movement / 50)  # Normalize to 50px
                factors.append(movement_score)
        
        # Calculate weighted average
        if factors:
            score = sum(factors) / len(factors)
        
        return score
    
    def detect_kickoff(self, player_positions: List[Tuple[float, float]], 
                       ball_xy: Optional[Tuple[float, float]],
                       frame_idx: int) -> Tuple[bool, float]:
        """
        Detect if current frame shows a kickoff.
        
        Kickoff indicators:
        1. Ball at center circle
        2. Two teams on opposite halves
        3. Players in formation (not clustered)
        
        Returns: (is_kickoff, confidence)
        """
        if ball_xy is None or len(player_positions) < 10:
            return False, 0.0
        
        confidence = 0.0
        
        # Check 1: Ball at center
        ball_dist_from_center = np.sqrt(
            (ball_xy[0] - self.center_x)**2 + 
            (ball_xy[1] - self.center_y)**2
        )
        
        if ball_dist_from_center < self.center_circle_radius:
            confidence += 0.4
        elif ball_dist_from_center < self.center_circle_radius * 2:
            confidence += 0.2
        else:
            return False, 0.0  # Ball too far from center
        
        # Check 2: Players on opposite halves
        left_half_count = sum(1 for p in player_positions if p[0] < self.center_x)
        right_half_count = len(player_positions) - left_half_count
        
        # Expect roughly equal split (8-12 on each side)
        if 7 <= left_half_count <= 13 and 7 <= right_half_count <= 13:
            balance = min(left_half_count, right_half_count) / max(left_half_count, right_half_count)
            confidence += 0.3 * balance
        
        # Check 3: Formation (players spread out, not clustered)
        if len(player_positions) >= 10:
            x_coords = [p[0] for p in player_positions]
            x_spread = (max(x_coords) - min(x_coords)) / self.frame_width
            
            if x_spread > 0.7:  # Players spread across 70%+ of pitch
                confidence += 0.3
            elif x_spread > 0.5:
                confidence += 0.15
        
        is_kickoff = confidence >= 0.65
        
        if is_kickoff:
            self.kickoff_candidates.append((frame_idx, confidence))
            self.kickoff_detected_count += 1
        
        return is_kickoff, confidence
    
    def vlm_verify_kickoff(self, frame_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Use VLM to verify kickoff detection.
        """
        if self.vlm_query is None:
            return False, 0.0
        
        prompt = """Analyze this football match frame.

Is this a KICKOFF moment at the start of a half?

KICKOFF indicators:
1. Ball is at CENTER CIRCLE
2. Two teams are on OPPOSITE HALVES
3. Players are in FORMATION (not clustered)
4. One player is about to kick the ball

Answer:
- "YES" if this is clearly a kickoff
- "NO" if this is regular play or a different situation

Simple answer: YES or NO"""
        
        try:
            response = self.vlm_query(frame_crop, prompt, max_tokens=10)
            response_upper = response.upper()
            
            is_kickoff = "YES" in response_upper
            confidence = 0.85 if is_kickoff else 0.15
            
            return is_kickoff, confidence
        except Exception as e:
            print(f"VLM kickoff verification error: {e}")
            return False, 0.0
    
    def update(self, frame_idx: int, player_positions: List[Tuple[float, float]],
               ball_xy: Optional[Tuple[float, float]], 
               prev_positions: Optional[Dict] = None,
               frame: Optional[np.ndarray] = None) -> Dict:
        """
        Update half detection with new frame data.
        
        Returns: Dictionary with current state and detected boundaries.
        """
        # === FAST PATH: If using pre-computed boundaries, skip detection ===
        if self.using_precomputed:
            # Just return current half based on frame index
            current_half = self.get_current_half(frame_idx)
            return {
                "current_half": current_half,
                "h1_start": self.h1_start,
                "h1_end": self.h1_end,
                "h2_start": self.h2_start,
                "h2_end": self.h2_end,
                "using_precomputed": True
            }
        
        # === SLOW PATH: Per-frame detection ===
        # Calculate activity score
        activity = self.calculate_activity_score(player_positions, ball_xy, prev_positions)
        self.activity_history.append(activity)
        self.total_activity_samples += 1
        
        # Get average activity over recent frames
        avg_activity = np.mean(list(self.activity_history)) if self.activity_history else 0
        
        # Detect state changes
        prev_state = self.current_state
        state_duration = frame_idx - self.state_start_frame
        
        # State machine for half detection
        if avg_activity >= self.playing_threshold:
            # High activity = playing
            if self.current_state != "playing":
                if state_duration >= self.min_state_duration_frames or self.current_state == "unknown":
                    # Check for kickoff (could be start of half)
                    is_kickoff, kickoff_conf = self.detect_kickoff(player_positions, ball_xy, frame_idx)
                    
                    # VLM verification if kickoff detected and frame available
                    if is_kickoff and frame is not None and self.vlm_query:
                        vlm_kickoff, vlm_conf = self.vlm_verify_kickoff(frame)
                        if vlm_kickoff:
                            kickoff_conf = max(kickoff_conf, vlm_conf)
                    
                    if is_kickoff and kickoff_conf >= 0.7:
                        # This is a half start!
                        if self.h1_start is None:
                            self.h1_start = frame_idx
                            print(f"🏁 1st Half START detected at frame {frame_idx} (conf={kickoff_conf:.2f})")
                        elif self.h1_end is not None and self.h2_start is None:
                            self.h2_start = frame_idx
                            print(f"🏁 2nd Half START detected at frame {frame_idx} (conf={kickoff_conf:.2f})")
                    
                    self.current_state = "playing"
                    self.state_start_frame = frame_idx
        
        elif avg_activity <= self.break_threshold:
            # Low activity = break or halftime
            if self.current_state == "playing":
                # Transition from playing to break
                if state_duration >= self.min_half_duration_frames:
                    # Long enough play duration - likely end of half
                    if self.h1_start is not None and self.h1_end is None:
                        self.h1_end = frame_idx
                        print(f"🏁 1st Half END detected at frame {frame_idx}")
                        self.current_state = "halftime"
                    elif self.h2_start is not None and self.h2_end is None:
                        self.h2_end = frame_idx
                        print(f"🏁 2nd Half END detected at frame {frame_idx}")
                        self.current_state = "postmatch"
                else:
                    self.current_state = "break"
                
                self.state_start_frame = frame_idx
            
            elif self.current_state == "break":
                # Check if break is long enough to be halftime
                if state_duration >= self.min_break_duration_frames:
                    if self.h1_end is not None and self.h2_start is None:
                        self.current_state = "halftime"
        
        return {
            "current_state": self.current_state,
            "activity_score": activity,
            "avg_activity": avg_activity,
            "h1_start": self.h1_start,
            "h1_end": self.h1_end,
            "h2_start": self.h2_start,
            "h2_end": self.h2_end,
            "kickoff_count": self.kickoff_detected_count
        }
    
    def get_half_timestamps(self, fps: float) -> Dict:
        """
        Get half boundaries in human-readable format.
        """
        def frame_to_time(frame):
            if frame is None:
                return None
            seconds = frame / fps
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
        
        return {
            "h1_start": {
                "frame": self.h1_start,
                "time": frame_to_time(self.h1_start)
            },
            "h1_end": {
                "frame": self.h1_end,
                "time": frame_to_time(self.h1_end)
            },
            "h2_start": {
                "frame": self.h2_start,
                "time": frame_to_time(self.h2_start)
            },
            "h2_end": {
                "frame": self.h2_end,
                "time": frame_to_time(self.h2_end)
            }
        }
    
    def get_current_half(self, frame_idx: int) -> str:
        """
        Determine which half a given frame belongs to.
        """
        if self.h1_start is not None:
            if self.h1_end is None or frame_idx < self.h1_end:
                if frame_idx >= self.h1_start:
                    return "first_half"
            elif self.h2_start is not None:
                if self.h2_end is None or frame_idx < self.h2_end:
                    if frame_idx >= self.h2_start:
                        return "second_half"
        
        return "unknown"


# === VLM PROMPTS FOR HALF DETECTION ===

KICKOFF_DETECTION_PROMPT = """Analyze this football match frame.

Is this a KICKOFF moment at the start of a half?

KICKOFF indicators:
1. Ball is at CENTER CIRCLE
2. Two teams are on OPPOSITE HALVES
3. Players are in FORMATION (not clustered)
4. One player is about to kick the ball

Answer:
- "YES" if this is clearly a kickoff
- "NO" if this is regular play or a different situation

Simple answer: YES or NO"""

HALFTIME_DETECTION_PROMPT = """Analyze this football match frame.

Is this HALFTIME or a BREAK in play?

HALFTIME indicators:
1. Players are LEAVING the pitch
2. Players are in the TUNNEL or DRESSING ROOM area
3. Camera showing EMPTY or NEARLY EMPTY pitch
4. Scoreboard or graphics showing "HALF TIME"

Answer:
- "YES" if this is clearly halftime
- "NO" if this is regular play or a brief stoppage

Simple answer: YES or NO"""
