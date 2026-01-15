"""
ScoutMe - Goalkeeper Save Detection Module
Detects goalkeeper saves after shot detection

CEO REQUIREMENT:
- shot attempt to save (GK Save detection)

LOGIC:
1. After a "Shot on target" is detected
2. Check if goalkeeper made a saving action
3. Classify save type: catch, parry, dive, tip over
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import cv2


class GKSaveDetector:
    """
    Detects goalkeeper saves following shot detection.
    Works in conjunction with ShotDetector.
    """
    
    def __init__(self, frame_width: int, frame_height: int, fps: float, vlm_query_func=None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.vlm_query = vlm_query_func
        
        # Save events
        self.save_events = []
        
        # Goalkeeper tracking
        self.gk_positions = {
            "left_goal": [],  # Recent GK positions near left goal
            "right_goal": []  # Recent GK positions near right goal
        }
        
        # Stats
        self.stats = {
            "Save - Catch": {"Team_A": 0, "Team_B": 0},
            "Save - Parry": {"Team_A": 0, "Team_B": 0},
            "Save - Dive": {"Team_A": 0, "Team_B": 0},
            "Save - Tip Over": {"Team_A": 0, "Team_B": 0},
            "Save - Other": {"Team_A": 0, "Team_B": 0}
        }
        
        # Goal zones (same as shot detection)
        self.left_goal_zone = (0, frame_width * 0.15)
        self.right_goal_zone = (frame_width * 0.85, frame_width)
        
        # Detection window (frames after shot to look for save)
        self.save_detection_window = int(fps * 2)  # 2 seconds after shot
    
    def find_goalkeeper(self, player_positions: Dict, player_teams: Dict,
                       target_goal: str, player_labels: Optional[Dict] = None) -> Optional[int]:
        """
        Find the goalkeeper near the target goal.
        
        Args:
            player_positions: Dict of tracker IDs to positions
            player_teams: Dict of tracker IDs to team names
            target_goal: "left_goal" or "right_goal"
            player_labels: Optional dict of tracker IDs to class IDs (0=ball, 1=gk, 2=player, 3=referee)
            
        Returns: Goalkeeper's tracker ID or None
        """
        goal_zone = self.left_goal_zone if target_goal == "left_goal" else self.right_goal_zone
        
        potential_gks = []
        
        for tid, pos in player_positions.items():
            if pos is None:
                continue
            
            x = pos[0] if hasattr(pos, '__len__') else pos
            
            # Check if player is in goal zone
            if goal_zone[0] <= x <= goal_zone[1]:
                # TEAM CHECK: Skip if identified as Referee
                team = player_teams.get(tid, "Unknown")
                if team == "Referee":
                    continue
                
                # LABEL CHECK: Skip if explicitly identified as Referee by YOLO
                if player_labels and player_labels.get(tid) == 3: # 3 is referee in football-player-detection.pt
                    continue
                
                # Score this candidate
                priority = 0
                if player_labels and player_labels.get(tid) == 1: # 1 is goalkeeper
                    priority = 100
                elif team in ["Blue", "White", "Team A", "Team B"]:
                    priority = 50
                else:
                    priority = 10 # Unknown team, but in goal zone
                
                # Distance to center of goal area (heuristics)
                goal_center = (goal_zone[0] + goal_zone[1]) / 2
                dist_to_center = abs(x - goal_center)
                score = priority - (dist_to_center / self.frame_width * 10)
                
                potential_gks.append((tid, score))
        
        if potential_gks:
            # Return tid with highest score
            potential_gks.sort(key=lambda x: x[1], reverse=True)
            return potential_gks[0][0]
        
        return None
    
    def detect_gk_action(self, frame: np.ndarray, gk_bbox: Tuple,
                        ball_xy: Optional[Tuple], prev_gk_bbox: Optional[Tuple] = None) -> Dict:
        """
        Detect goalkeeper action based on movement and position.
        
        Returns: {"action_type": str, "confidence": float, "metrics": dict}
        """
        if gk_bbox is None:
            return {"action_type": None, "confidence": 0, "metrics": {}}
        
        x1, y1, x2, y2 = gk_bbox
        gk_height = y2 - y1
        gk_width = x2 - x1
        gk_center_y = (y1 + y2) / 2
        gk_center_x = (x1 + x2) / 2
        
        action_type = None
        confidence = 0.0
        metrics = {
            "gk_height": gk_height,
            "gk_width": gk_width,
            "gk_center": (gk_center_x, gk_center_y)
        }
        
        # Detect significant movement if previous bbox available
        if prev_gk_bbox is not None:
            px1, py1, px2, py2 = prev_gk_bbox
            prev_center_x = (px1 + px2) / 2
            prev_center_y = (py1 + py2) / 2
            
            # Calculate movement
            dx = gk_center_x - prev_center_x
            dy = gk_center_y - prev_center_y
            movement = np.sqrt(dx**2 + dy**2)
            
            metrics["movement"] = movement
            metrics["dx"] = dx
            metrics["dy"] = dy
            
            # Detect dive (significant horizontal movement)
            if abs(dx) > gk_width * 0.5:  # Moved more than half body width
                action_type = "dive"
                confidence = min(90, 50 + abs(dx) / gk_width * 40)
            
            # Detect jump (significant upward movement)
            elif dy < -gk_height * 0.2:  # Moved up more than 20% of height
                action_type = "jump"
                confidence = min(85, 50 + abs(dy) / gk_height * 40)
        
        # Check ball position relative to goalkeeper
        if ball_xy is not None:
            ball_dist = np.sqrt((ball_xy[0] - gk_center_x)**2 + (ball_xy[1] - gk_center_y)**2)
            metrics["ball_distance"] = ball_dist
            
            # Ball very close to GK = likely catch or parry
            if ball_dist < gk_height * 0.5:
                if action_type is None:
                    action_type = "catch"
                    confidence = 75
        
        return {"action_type": action_type, "confidence": confidence, "metrics": metrics}
    
    def vlm_detect_save(self, frame_crop: np.ndarray, shot_type: str) -> Tuple[bool, str, float]:
        """
        Use VLM to detect and classify goalkeeper save.
        
        Returns: (is_save, save_type, confidence)
        """
        if self.vlm_query is None:
            return False, None, 0.0
        
        prompt = f"""Analyze this football match frame for a GOALKEEPER SAVE.

CONTEXT:
- A shot has just been taken ({shot_type})
- Focus on the GOALKEEPER's action

TASK: Did the goalkeeper make a SAVE?

SAVE TYPES (choose one if applicable):
1. CATCH - Goalkeeper catches and holds the ball
2. PARRY - Goalkeeper punches/deflects ball away with fists
3. DIVE - Goalkeeper dives to stop the ball
4. TIP OVER - Goalkeeper tips the ball over the crossbar
5. BLOCK - Goalkeeper uses body to block (not diving)

SAVE INDICATORS:
- Goalkeeper in unusual position (diving, stretching)
- Ball direction changed after GK contact
- Ball held by goalkeeper
- Ball deflected away from goal

NOT A SAVE:
- Ball went wide/over without GK touch
- Ball went in (goal scored)
- Outfield player blocked the shot

Answer format:
SAVE: YES or NO
TYPE: [save type] or NONE
CONFIDENCE: [0-100]"""
        
        try:
            response = self.vlm_query(frame_crop, prompt, max_tokens=50)
            response_upper = response.upper()
            
            # Parse response
            is_save = "SAVE: YES" in response_upper or ("YES" in response_upper and "NO" not in response_upper[:20])
            
            save_type = None
            if "CATCH" in response_upper:
                save_type = "Save - Catch"
            elif "PARRY" in response_upper:
                save_type = "Save - Parry"
            elif "DIVE" in response_upper:
                save_type = "Save - Dive"
            elif "TIP" in response_upper:
                save_type = "Save - Tip Over"
            elif "BLOCK" in response_upper:
                save_type = "Save - Other"
            elif is_save:
                save_type = "Save - Other"
            
            # Extract confidence
            confidence = 75 if is_save else 20
            if "CONFIDENCE:" in response_upper:
                try:
                    conf_str = response_upper.split("CONFIDENCE:")[1][:10]
                    conf_num = int(''.join(c for c in conf_str if c.isdigit())[:3])
                    confidence = min(100, max(0, conf_num))
                except:
                    pass
            
            return is_save, save_type, confidence
            
        except Exception as e:
            print(f"VLM save detection error: {e}")
            return False, None, 0.0
    
    def detect_save_after_shot(self, frame: np.ndarray, frame_idx: int,
                               shot_event: Dict, player_positions: Dict,
                               player_teams: Dict, player_bboxes: Dict,
                               ball_xy: Optional[Tuple] = None,
                               frame_buffer: Optional[List] = None,
                               player_labels: Optional[Dict] = None) -> Optional[Dict]:
        """
        Detect if a save occurred after a shot.
        
        Args:
            frame: Current video frame
            frame_idx: Current frame index
            shot_event: The shot event that triggered this check
            player_positions: Current player positions
            player_teams: Player team assignments
            player_bboxes: Player bounding boxes
            ball_xy: Current ball position
            frame_buffer: Optional list of recent frames for analysis
            player_labels: Optional dict of tracker IDs to class IDs
        
        Returns: Save event dict or None
        """
        shot_type = shot_event.get("shot_type", "Shot on target")
        target_goal = shot_event.get("target_goal", "right_goal")
        shot_frame = shot_event.get("frame", frame_idx)
        
        # Only check for saves on "Shot on target"
        if shot_type != "Shot on target":
            return None
        
        # Check if we're within the detection window
        frames_since_shot = frame_idx - shot_frame
        if frames_since_shot > self.save_detection_window:
            return None
        
        # Find goalkeeper near target goal
        gk_tid = self.find_goalkeeper(player_positions, player_teams, target_goal, player_labels)
        
        if gk_tid is None:
            return None
        
        gk_bbox = player_bboxes.get(gk_tid)
        gk_team = player_teams.get(gk_tid, "Unknown")
        
        # Get frame crop around goalkeeper
        if gk_bbox is not None:
            x1, y1, x2, y2 = map(int, gk_bbox)
            crop_margin = 100
            crop_x1 = max(0, x1 - crop_margin)
            crop_y1 = max(0, y1 - crop_margin)
            crop_x2 = min(self.frame_width, x2 + crop_margin)
            crop_y2 = min(self.frame_height, y2 + crop_margin)
            
            gk_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if gk_crop.size > 0:
                # Use VLM to detect save
                is_save, save_type, confidence = self.vlm_detect_save(gk_crop, shot_type)
                
                if is_save and confidence >= 60:
                    # Create save event
                    save_event = {
                        "time": self._format_time(frame_idx / self.fps),
                        "frame": frame_idx,
                        "shot_frame": shot_frame,
                        "goalkeeper_id": int(gk_tid),
                        "goalkeeper_team": gk_team,
                        "save_type": save_type,
                        "confidence": confidence,
                        "gk_x": round((x1 + x2) / 2, 1),
                        "gk_y": round((y1 + y2) / 2, 1),
                        "ball_x": round(ball_xy[0], 1) if ball_xy is not None else round((x1 + x2) / 2, 1),
                        "ball_y": round(ball_xy[1], 1) if ball_xy is not None else round((y1 + y2) / 2, 1),
                        "target_goal": target_goal,
                        "related_shot": shot_event.get("time")
                    }
                    
                    # Add to events
                    self.save_events.append(save_event)
                    
                    # Update stats
                    if save_type and save_type in self.stats:
                        stat_team = "Team_A" if gk_team in ["Blue", "Team A"] else "Team_B"
                        if stat_team in self.stats[save_type]:
                            self.stats[save_type][stat_team] += 1
                    
                    print(f"  🧤 [SAVE DETECTED] Frame {frame_idx}: {save_type} by GK #{gk_tid} ({gk_team}) | conf={confidence}%")
                    
                    return save_event
        
        return None
    
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def get_save_events(self) -> List[Dict]:
        """Get all recorded save events"""
        return self.save_events
    
    def get_stats(self) -> Dict:
        """Get save statistics"""
        return self.stats
    
    def print_summary(self):
        """Print save detection summary"""
        total_saves = len(self.save_events)
        
        if total_saves == 0:
            print("\n🧤 No goalkeeper saves detected")
            return
        
        print("\n" + "=" * 70)
        print("🧤 GOALKEEPER SAVE SUMMARY")
        print("=" * 70)
        
        print(f"\nSave Type           | Team A    | Team B    | Total")
        print("-" * 60)
        
        for save_type in self.stats:
            team_a = self.stats[save_type].get("Team_A", 0)
            team_b = self.stats[save_type].get("Team_B", 0)
            total = team_a + team_b
            if total > 0:
                print(f"{save_type:<20} | {team_a:^9} | {team_b:^9} | {total:^5}")
        
        print("-" * 60)
        print(f"{'TOTAL':<20} | {sum(self.stats[st].get('Team_A', 0) for st in self.stats):^9} | {sum(self.stats[st].get('Team_B', 0) for st in self.stats):^9} | {total_saves:^5}")
        print("=" * 70)


def generate_save_report_json(save_events: List[Dict], stats: Dict, output_path: str = "save_report.json") -> str:
    """Generate JSON report of goalkeeper saves"""
    import json
    
    report = {
        "save_events": save_events,
        "statistics": stats,
        "total_saves": len(save_events)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Save report saved to: {output_path}")
    return output_path


# === VLM PROMPTS ===

GK_SAVE_PROMPT = """Analyze this football match frame for a GOALKEEPER SAVE.

CONTEXT:
- A shot has just been taken toward the goal
- Focus on the GOALKEEPER's action

TASK: Did the goalkeeper make a SAVE?

SAVE TYPES (choose one if applicable):
1. CATCH - Goalkeeper catches and holds the ball
2. PARRY - Goalkeeper punches/deflects ball away with fists
3. DIVE - Goalkeeper dives to stop the ball
4. TIP OVER - Goalkeeper tips the ball over the crossbar
5. BLOCK - Goalkeeper uses body to block (not diving)

SAVE INDICATORS:
- Goalkeeper in unusual position (diving, stretching)
- Ball direction changed after GK contact
- Ball held by goalkeeper
- Ball deflected away from goal

NOT A SAVE:
- Ball went wide/over without GK touch
- Ball went in (goal scored)
- Outfield player blocked the shot

Answer format:
SAVE: YES or NO
TYPE: [save type] or NONE
CONFIDENCE: [0-100]"""
