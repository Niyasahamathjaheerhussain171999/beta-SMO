"""
ScoutMe - Team Direction Detection Module
Automatically detects which team is attacking which direction

CEO REQUIREMENT:
- Direction of player from each team - defending and attacking for 1st half
- Direction of player from each team - defending and attacking for 2nd half

LOGIC (from pitch diagram):
- First Half: Team 1 defends LEFT goal, attacks RIGHT
            Team 2 defends RIGHT goal, attacks LEFT
- Second Half: Teams SWAP sides
            Team 1 defends RIGHT goal, attacks LEFT
            Team 2 defends LEFT goal, attacks RIGHT

TECH LEAD APPROVED:
- GEOMETRY-FIRST approach (100x faster than VLM)
- Uses Average X position of each team to determine direction
- VLM only used for sanity check (optional)
- Supports pre-computed directions from MatchSegmenter
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple, List
import cv2


class TeamDirectionDetector:
    """
    Detects team attack/defend directions by analyzing:
    1. GEOMETRY-FIRST: Average X position (fast, 100x faster than VLM)
    2. Goalkeeper positions (GK defends their goal)
    3. Team spread patterns (attacking team spreads toward goal)
    4. VLM verification for confirmation (optional, single call)
    
    Supports pre-computed directions from MatchSegmenter.
    """
    
    def __init__(self, frame_width: int, frame_height: int, vlm_query_func=None,
                 precomputed_directions: Optional[Dict] = None):
        """
        Initialize Team Direction Detector.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            vlm_query_func: Optional VLM function (only used for sanity check)
            precomputed_directions: Optional dict from MatchSegmenter
                                   {"first_half": {"team_a": "right", "team_b": "left"}, ...}
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.vlm_query = vlm_query_func
        
        # Flag to indicate if using pre-computed directions
        self.using_precomputed = False
        
        # Team direction state
        self.team_directions = {
            "first_half": {
                "team_1": {"defending_goal": None, "attacking_direction": None},
                "team_2": {"defending_goal": None, "attacking_direction": None}
            },
            "second_half": {
                "team_1": {"defending_goal": None, "attacking_direction": None},
                "team_2": {"defending_goal": None, "attacking_direction": None}
            }
        }
        
        # Apply pre-computed directions if provided
        if precomputed_directions:
            self._apply_precomputed_directions(precomputed_directions)
        
        # Goalkeeper tracking
        self.gk_positions = {
            "left_goal": [],  # (team, frame_idx, confidence)
            "right_goal": []
        }
        
        # Team color mappings
        self.team_colors = {
            "team_1": None,  # Will be detected (e.g., "Blue")
            "team_2": None   # Will be detected (e.g., "White")
        }
        
        # Detection confidence
        self.detection_confidence = 0.0
        self.samples_collected = 0
        
        # Goal zones
        self.left_goal_zone = (0, frame_width * 0.15)  # Left 15%
        self.right_goal_zone = (frame_width * 0.85, frame_width)  # Right 15%
        
        # Geometry-based detection state
        self.team_avg_x_history = {"team_a": [], "team_b": []}
        self.geometry_locked = False  # Once confident, lock the result
    
    def _apply_precomputed_directions(self, directions: Dict):
        """Apply pre-computed directions from MatchSegmenter."""
        first_half = directions.get("first_half", {})
        second_half = directions.get("second_half", {})
        
        # Map to internal format
        if first_half.get("team_a"):
            self.team_directions["first_half"]["team_1"]["attacking_direction"] = first_half["team_a"]
            self.team_directions["first_half"]["team_1"]["defending_goal"] = "left" if first_half["team_a"] == "right" else "right"
        
        if first_half.get("team_b"):
            self.team_directions["first_half"]["team_2"]["attacking_direction"] = first_half["team_b"]
            self.team_directions["first_half"]["team_2"]["defending_goal"] = "left" if first_half["team_b"] == "right" else "right"
        
        if second_half.get("team_a"):
            self.team_directions["second_half"]["team_1"]["attacking_direction"] = second_half["team_a"]
            self.team_directions["second_half"]["team_1"]["defending_goal"] = "left" if second_half["team_a"] == "right" else "right"
        
        if second_half.get("team_b"):
            self.team_directions["second_half"]["team_2"]["attacking_direction"] = second_half["team_b"]
            self.team_directions["second_half"]["team_2"]["defending_goal"] = "left" if second_half["team_b"] == "right" else "right"
        
        self.using_precomputed = True
        self.geometry_locked = True
        self.detection_confidence = 1.0
        
        print(f"📋 TeamDirectionDetector: Using pre-computed directions")
        print(f"   First Half: Team A → {first_half.get('team_a')}, Team B → {first_half.get('team_b')}")
        print(f"   Second Half: Team A → {second_half.get('team_a')}, Team B → {second_half.get('team_b')}")
    
    def set_directions(self, first_half_team_a: str = None, first_half_team_b: str = None,
                      second_half_team_a: str = None, second_half_team_b: str = None):
        """
        Manually set team directions (e.g., from MatchSegmenter).
        
        Args:
            first_half_team_a: "left" or "right" - direction Team A attacks in H1
            first_half_team_b: "left" or "right" - direction Team B attacks in H1
            second_half_team_a: "left" or "right" - direction Team A attacks in H2
            second_half_team_b: "left" or "right" - direction Team B attacks in H2
        """
        directions = {
            "first_half": {"team_a": first_half_team_a, "team_b": first_half_team_b},
            "second_half": {"team_a": second_half_team_a, "team_b": second_half_team_b}
        }
        self._apply_precomputed_directions(directions)
    
    def detect_direction_geometry(self, player_positions: Dict, player_teams: Dict,
                                  team_a_name: str, team_b_name: str) -> Dict:
        """
        GEOMETRY-FIRST: Detect team directions using average X position.
        This is 100x faster than VLM and mathematically precise.
        
        Logic:
        - Team with lower avg X is on LEFT side → attacks RIGHT
        - Team with higher avg X is on RIGHT side → attacks LEFT
        
        Returns: {"team_a": "left"/"right", "team_b": "left"/"right", "confidence": float}
        """
        team_a_x = []
        team_b_x = []
        
        for tid, pos in player_positions.items():
            if pos is None:
                continue
            
            team = player_teams.get(tid, "Unknown")
            x = pos[0] if hasattr(pos, '__len__') else pos
            
            if team == team_a_name:
                team_a_x.append(x)
            elif team == team_b_name:
                team_b_x.append(x)
        
        if len(team_a_x) < 3 or len(team_b_x) < 3:
            return {"team_a": None, "team_b": None, "confidence": 0.0}
        
        avg_x_a = np.mean(team_a_x)
        avg_x_b = np.mean(team_b_x)
        
        # Store for history
        self.team_avg_x_history["team_a"].append(avg_x_a)
        self.team_avg_x_history["team_b"].append(avg_x_b)
        
        # Keep only last 50 samples
        if len(self.team_avg_x_history["team_a"]) > 50:
            self.team_avg_x_history["team_a"] = self.team_avg_x_history["team_a"][-50:]
            self.team_avg_x_history["team_b"] = self.team_avg_x_history["team_b"][-50:]
        
        # Calculate stable average
        stable_avg_a = np.mean(self.team_avg_x_history["team_a"])
        stable_avg_b = np.mean(self.team_avg_x_history["team_b"])
        
        center_x = self.frame_width / 2
        
        # Determine directions
        if stable_avg_a < stable_avg_b:
            # Team A is on LEFT → attacks RIGHT
            team_a_direction = "right"
            team_b_direction = "left"
        else:
            # Team A is on RIGHT → attacks LEFT
            team_a_direction = "left"
            team_b_direction = "right"
        
        # Calculate confidence based on separation
        separation = abs(stable_avg_a - stable_avg_b) / self.frame_width
        confidence = min(1.0, separation * 3)  # Full confidence at 33% separation
        
        return {
            "team_a": team_a_direction,
            "team_b": team_b_direction,
            "confidence": confidence,
            "avg_x_a": stable_avg_a,
            "avg_x_b": stable_avg_b
        }
    
    def detect_goalkeeper(self, player_positions: Dict, player_teams: Dict, 
                         player_bboxes: Dict) -> Dict[str, Optional[int]]:
        """
        Detect likely goalkeepers based on position (in goal area).
        
        Returns: {"left_goal": tid or None, "right_goal": tid or None}
        """
        goalkeepers = {"left_goal": None, "right_goal": None}
        
        for tid, pos in player_positions.items():
            if pos is None:
                continue
            
            x = pos[0] if hasattr(pos, '__len__') else pos
            
            # Check left goal zone
            if self.left_goal_zone[0] <= x <= self.left_goal_zone[1]:
                if goalkeepers["left_goal"] is None:
                    goalkeepers["left_goal"] = tid
            
            # Check right goal zone
            elif self.right_goal_zone[0] <= x <= self.right_goal_zone[1]:
                if goalkeepers["right_goal"] is None:
                    goalkeepers["right_goal"] = tid
        
        return goalkeepers
    
    def analyze_team_spread(self, player_positions: Dict, player_teams: Dict,
                           team_a_name: str, team_b_name: str) -> Dict:
        """
        Analyze team spread to determine attack direction.
        
        Attacking team tends to:
        - Have more players in opponent's half
        - Have higher average x-position toward attack
        
        Returns: {"team_a_avg_x": float, "team_b_avg_x": float}
        """
        team_a_positions = []
        team_b_positions = []
        
        for tid, pos in player_positions.items():
            if pos is None:
                continue
            
            team = player_teams.get(tid, "Unknown")
            x = pos[0] if hasattr(pos, '__len__') else pos
            
            if team == team_a_name:
                team_a_positions.append(x)
            elif team == team_b_name:
                team_b_positions.append(x)
        
        result = {
            "team_a_avg_x": np.mean(team_a_positions) if team_a_positions else self.frame_width / 2,
            "team_b_avg_x": np.mean(team_b_positions) if team_b_positions else self.frame_width / 2,
            "team_a_count": len(team_a_positions),
            "team_b_count": len(team_b_positions)
        }
        
        return result
    
    def update(self, frame_idx: int, player_positions: Dict, player_teams: Dict,
               player_bboxes: Dict, team_a_name: str, team_b_name: str,
               current_half: str = "first_half") -> Dict:
        """
        Update team direction detection with new frame data.
        
        Returns: Current team directions and confidence.
        """
        self.samples_collected += 1
        
        # Store team colors
        if self.team_colors["team_1"] is None:
            self.team_colors["team_1"] = team_a_name
            self.team_colors["team_2"] = team_b_name
        
        # === FAST PATH: If using pre-computed directions, skip detection ===
        if self.using_precomputed and self.geometry_locked:
            return {
                "team_directions": self.team_directions,
                "current_half": current_half,
                "confidence": self.detection_confidence,
                "using_precomputed": True,
                "samples": self.samples_collected
            }
        
        # === GEOMETRY-FIRST: Use average X position (100x faster than VLM) ===
        if not self.geometry_locked:
            geo_result = self.detect_direction_geometry(player_positions, player_teams, team_a_name, team_b_name)
            
            # Lock geometry result once confident (after ~20 samples)
            if geo_result["confidence"] >= 0.7 and self.samples_collected >= 20:
                half_key = current_half
                
                if geo_result["team_a"] == "right":
                    # Team A attacks right → defends left
                    self.team_directions[half_key]["team_1"] = {
                        "team_color": team_a_name,
                        "defending_goal": "left",
                        "attacking_direction": "left_to_right"
                    }
                    self.team_directions[half_key]["team_2"] = {
                        "team_color": team_b_name,
                        "defending_goal": "right",
                        "attacking_direction": "right_to_left"
                    }
                else:
                    # Team A attacks left → defends right
                    self.team_directions[half_key]["team_1"] = {
                        "team_color": team_a_name,
                        "defending_goal": "right",
                        "attacking_direction": "right_to_left"
                    }
                    self.team_directions[half_key]["team_2"] = {
                        "team_color": team_b_name,
                        "defending_goal": "left",
                        "attacking_direction": "left_to_right"
                    }
                
                self.detection_confidence = geo_result["confidence"]
                self.geometry_locked = True
                print(f"🎯 Team directions LOCKED via geometry (conf={geo_result['confidence']:.2f})")
                print(f"   {team_a_name} avg X: {geo_result['avg_x_a']:.0f} → attacks {geo_result['team_a']}")
                print(f"   {team_b_name} avg X: {geo_result['avg_x_b']:.0f} → attacks {geo_result['team_b']}")
        
        # === FALLBACK: GK position detection (slower but reliable) ===
        # Detect goalkeepers
        goalkeepers = self.detect_goalkeeper(player_positions, player_teams, player_bboxes)
        
        # Determine GK teams
        left_gk_team = player_teams.get(goalkeepers["left_goal"], "Unknown") if goalkeepers["left_goal"] else None
        right_gk_team = player_teams.get(goalkeepers["right_goal"], "Unknown") if goalkeepers["right_goal"] else None
        
        # Record GK positions for voting
        if left_gk_team and left_gk_team not in ["Unknown", "Referee"]:
            self.gk_positions["left_goal"].append((left_gk_team, frame_idx, 0.8))
        if right_gk_team and right_gk_team not in ["Unknown", "Referee"]:
            self.gk_positions["right_goal"].append((right_gk_team, frame_idx, 0.8))
        
        # Analyze team spread
        spread = self.analyze_team_spread(player_positions, player_teams, team_a_name, team_b_name)
        
        # Determine directions based on GK positions (most reliable)
        if len(self.gk_positions["left_goal"]) >= 10 and len(self.gk_positions["right_goal"]) >= 10:
            # Get majority team for each goal
            left_teams = [t[0] for t in self.gk_positions["left_goal"][-50:]]  # Last 50 samples
            right_teams = [t[0] for t in self.gk_positions["right_goal"][-50:]]
            
            from collections import Counter
            left_majority = Counter(left_teams).most_common(1)
            right_majority = Counter(right_teams).most_common(1)
            
            if left_majority and right_majority:
                left_gk_team = left_majority[0][0]
                right_gk_team = right_majority[0][0]
                
                # GK defends their goal, attacks opposite
                half_key = current_half
                
                if left_gk_team == team_a_name:
                    # Team A GK at left → Team A defends left, attacks right
                    self.team_directions[half_key]["team_1"] = {
                        "team_color": team_a_name,
                        "defending_goal": "left",
                        "attacking_direction": "left_to_right"
                    }
                    self.team_directions[half_key]["team_2"] = {
                        "team_color": team_b_name,
                        "defending_goal": "right",
                        "attacking_direction": "right_to_left"
                    }
                    self.detection_confidence = 0.9
                
                elif left_gk_team == team_b_name:
                    # Team B GK at left → Team B defends left, attacks right
                    self.team_directions[half_key]["team_1"] = {
                        "team_color": team_a_name,
                        "defending_goal": "right",
                        "attacking_direction": "right_to_left"
                    }
                    self.team_directions[half_key]["team_2"] = {
                        "team_color": team_b_name,
                        "defending_goal": "left",
                        "attacking_direction": "left_to_right"
                    }
                    self.detection_confidence = 0.9
        
        return {
            "team_directions": self.team_directions,
            "current_half": current_half,
            "confidence": self.detection_confidence,
            "goalkeepers": goalkeepers,
            "team_spread": spread,
            "samples": self.samples_collected
        }
    
    def swap_for_second_half(self):
        """
        Swap team directions for second half (teams change ends).
        """
        first_half = self.team_directions["first_half"]
        
        # Swap directions for second half
        self.team_directions["second_half"]["team_1"] = {
            "team_color": first_half["team_1"].get("team_color"),
            "defending_goal": "right" if first_half["team_1"].get("defending_goal") == "left" else "left",
            "attacking_direction": "right_to_left" if first_half["team_1"].get("attacking_direction") == "left_to_right" else "left_to_right"
        }
        
        self.team_directions["second_half"]["team_2"] = {
            "team_color": first_half["team_2"].get("team_color"),
            "defending_goal": "right" if first_half["team_2"].get("defending_goal") == "left" else "left",
            "attacking_direction": "right_to_left" if first_half["team_2"].get("attacking_direction") == "left_to_right" else "left_to_right"
        }
        
        print(f"🔄 Teams swapped for 2nd half:")
        print(f"   Team 1 ({self.team_colors['team_1']}): Defends {self.team_directions['second_half']['team_1']['defending_goal']}, Attacks {self.team_directions['second_half']['team_1']['attacking_direction']}")
        print(f"   Team 2 ({self.team_colors['team_2']}): Defends {self.team_directions['second_half']['team_2']['defending_goal']}, Attacks {self.team_directions['second_half']['team_2']['attacking_direction']}")
    
    def vlm_verify_directions(self, frame_crop: np.ndarray, team_a_name: str, 
                              team_b_name: str) -> Dict:
        """
        Use VLM to verify team directions.
        """
        if self.vlm_query is None:
            return {"verified": False}
        
        prompt = f"""Analyze this football match frame to determine team directions.

TASK: Identify which team is attacking which goal.

ANALYZE:
1. Find the TWO GOALKEEPERS (players near each goal)
2. Identify their jersey colors
3. Determine which goal each team DEFENDS

Teams in this match:
- Team 1: {team_a_name} jerseys
- Team 2: {team_b_name} jerseys

RULES:
- Goalkeeper DEFENDS their goal (stands near it)
- Team ATTACKS the opposite goal

OUTPUT FORMAT (answer exactly):
Team defending LEFT goal: [color]
Team defending RIGHT goal: [color]"""
        
        try:
            response = self.vlm_query(frame_crop, prompt, max_tokens=50)
            
            # Parse response
            result = {
                "verified": True,
                "raw_response": response
            }
            
            response_upper = response.upper()
            
            if "LEFT" in response_upper and team_a_name.upper() in response_upper.split("LEFT")[1][:50]:
                result["left_defending"] = team_a_name
            elif "LEFT" in response_upper and team_b_name.upper() in response_upper.split("LEFT")[1][:50]:
                result["left_defending"] = team_b_name
            
            if "RIGHT" in response_upper and team_a_name.upper() in response_upper.split("RIGHT")[1][:50]:
                result["right_defending"] = team_a_name
            elif "RIGHT" in response_upper and team_b_name.upper() in response_upper.split("RIGHT")[1][:50]:
                result["right_defending"] = team_b_name
            
            return result
            
        except Exception as e:
            print(f"VLM direction verification error: {e}")
            return {"verified": False, "error": str(e)}
    
    def get_attack_direction(self, team_color: str, current_half: str) -> Optional[str]:
        """
        Get attack direction for a specific team in the current half.
        
        Returns: "left_to_right" or "right_to_left" or None
        """
        half = self.team_directions.get(current_half, {})
        
        for team_key in ["team_1", "team_2"]:
            team_info = half.get(team_key, {})
            if team_info.get("team_color") == team_color:
                return team_info.get("attacking_direction")
        
        return None
    
    def get_defending_goal(self, team_color: str, current_half: str) -> Optional[str]:
        """
        Get defending goal for a specific team in the current half.
        
        Returns: "left" or "right" or None
        """
        half = self.team_directions.get(current_half, {})
        
        for team_key in ["team_1", "team_2"]:
            team_info = half.get(team_key, {})
            if team_info.get("team_color") == team_color:
                return team_info.get("defending_goal")
        
        return None
    
    def get_directions_json(self) -> Dict:
        """
        Get team directions in JSON format for output.
        """
        return {
            "first_half": {
                "team_1": self.team_directions["first_half"]["team_1"],
                "team_2": self.team_directions["first_half"]["team_2"]
            },
            "second_half": {
                "team_1": self.team_directions["second_half"]["team_1"],
                "team_2": self.team_directions["second_half"]["team_2"]
            },
            "confidence": self.detection_confidence,
            "samples_analyzed": self.samples_collected
        }


# === VLM PROMPTS FOR TEAM DIRECTION ===

TEAM_DIRECTION_PROMPT = """Analyze this football match frame to determine team attack directions.

ANALYZE:
1. Identify the TWO teams by jersey color
2. Find GOALKEEPERS (players in goal area, different jersey)
3. Determine which goal each team is DEFENDING

RULES:
- Team defends the goal where THEIR goalkeeper stands
- In 2nd half, teams SWAP sides from 1st half

OUTPUT FORMAT (JSON only):
{
  "team_1_color": "blue/white/red/...",
  "team_1_defending_goal": "left" | "right",
  "team_1_attacking_direction": "left_to_right" | "right_to_left",
  "team_2_color": "...",
  "team_2_defending_goal": "left" | "right",
  "team_2_attacking_direction": "left_to_right" | "right_to_left",
  "goalkeeper_1_position": "left_goal" | "right_goal",
  "goalkeeper_2_position": "left_goal" | "right_goal",
  "confidence": 0-100
}"""
