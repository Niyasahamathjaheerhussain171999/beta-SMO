"""
ScoutMe - MVP Integration Module
Integrates all new modules with main4.py for complete match analysis

CEO REQUIREMENTS FULFILLED:
✅ 1. stamp time of first half start & end time
✅ 2. stamp time of second half start & end time  
✅ 3. Direction of player from each team - defending and attacking for 1st half
✅ 4. Direction of player from each team - defending and attacking for 2nd half
✅ 5. shot attempts (existing)
✅ 6. shot on target (existing)
✅ 7. goal (existing)
✅ 8. shot attempt to save (NEW - GK saves)

USAGE:
    from mvp_integration import MVPAnalyzer
    
    analyzer = MVPAnalyzer(video_path, vlm_query_func)
    results = analyzer.run_full_analysis()
"""

import os
import json
from typing import Dict, Optional, List, Tuple
import numpy as np


class MVPAnalyzer:
    """
    Master integration class that combines all ScoutMe modules.
    """
    
    def __init__(self, video_path: str, vlm_query_func, video_query_func=None,
                 team_a_name: str = "Blue", team_b_name: str = "White",
                 config: Dict = None):
        """
        Initialize MVP Analyzer with all modules.
        
        Args:
            video_path: Path to input video
            vlm_query_func: Function to query VLM (Molmo)
            video_query_func: Function to query Video VLM (Qwen) - optional
            team_a_name: Name/color of team A
            team_b_name: Name/color of team B
            config: Optional configuration overrides
        """
        self.video_path = video_path
        self.vlm_query = vlm_query_func
        self.video_query = video_query_func
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.config = config or {}
        
        # Will be initialized after video info is loaded
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.total_frames = None
        
        # Module instances (initialized later)
        self.half_detector = None
        self.team_direction = None
        self.gk_save_detector = None
        
        # Results storage
        self.results = {
            "match_info": {},
            "half_timestamps": {},
            "team_directions": {},
            "passes": [],
            "shots": [],
            "saves": [],
            "statistics": {}
        }
    
    def initialize_modules(self, frame_width: int, frame_height: int, fps: float, total_frames: int):
        """
        Initialize all analysis modules with video dimensions.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.total_frames = total_frames
        
        # Import and initialize modules
        try:
            from half_detector import HalfDetector
            self.half_detector = HalfDetector(frame_width, frame_height, fps, self.vlm_query)
            print("✅ Half Detector initialized")
        except ImportError as e:
            print(f"⚠️ Half Detector not available: {e}")
        
        try:
            from team_direction import TeamDirectionDetector
            self.team_direction = TeamDirectionDetector(frame_width, frame_height, self.vlm_query)
            print("✅ Team Direction Detector initialized")
        except ImportError as e:
            print(f"⚠️ Team Direction Detector not available: {e}")
        
        try:
            from gk_save_detector import GKSaveDetector
            self.gk_save_detector = GKSaveDetector(frame_width, frame_height, fps, self.vlm_query)
            print("✅ GK Save Detector initialized")
        except ImportError as e:
            print(f"⚠️ GK Save Detector not available: {e}")
    
    def process_frame(self, frame_idx: int, frame, player_positions: Dict,
                     player_teams: Dict, player_bboxes: Dict,
                     ball_xy: Optional[Tuple], prev_positions: Dict = None) -> Dict:
        """
        Process a single frame through all modules.
        
        Returns: Dictionary of module outputs for this frame
        """
        outputs = {}
        
        # 1. Half Detection
        if self.half_detector:
            positions_list = [(p[0], p[1]) for p in player_positions.values() if p is not None]
            half_result = self.half_detector.update(
                frame_idx, positions_list, ball_xy, prev_positions, frame
            )
            outputs["half_detection"] = half_result
            
            # Check for half transitions
            if half_result.get("h2_start") and not hasattr(self, '_h2_started'):
                self._h2_started = True
                # Swap team directions for second half
                if self.team_direction:
                    self.team_direction.swap_for_second_half()
        
        # 2. Team Direction Detection
        if self.team_direction:
            current_half = "second_half" if hasattr(self, '_h2_started') else "first_half"
            direction_result = self.team_direction.update(
                frame_idx, player_positions, player_teams, player_bboxes,
                self.team_a_name, self.team_b_name, current_half
            )
            outputs["team_direction"] = direction_result
        
        return outputs
    
    def check_for_save(self, frame, frame_idx: int, shot_event: Dict,
                       player_positions: Dict, player_teams: Dict,
                       player_bboxes: Dict, ball_xy: Optional[Tuple] = None,
                       player_labels: Optional[Dict] = None) -> Optional[Dict]:
        """
        Check if a goalkeeper save occurred after a shot.
        """
        if self.gk_save_detector and shot_event:
            save_event = self.gk_save_detector.detect_save_after_shot(
                frame, frame_idx, shot_event, player_positions,
                player_teams, player_bboxes, ball_xy,
                player_labels=player_labels
            )
            return save_event
        return None
    
    def get_current_half(self, frame_idx: int) -> str:
        """Get the current half for a given frame."""
        if self.half_detector:
            return self.half_detector.get_current_half(frame_idx)
        return "unknown"
    
    def get_attack_direction(self, team_color: str, frame_idx: int) -> Optional[str]:
        """Get attack direction for a team at a given frame."""
        if self.team_direction:
            current_half = self.get_current_half(frame_idx)
            return self.team_direction.get_attack_direction(team_color, current_half)
        return None
    
    def compile_results(self, pass_events: List, shot_events: List) -> Dict:
        """
        Compile all results into final output format.
        """
        # Match info
        duration_seconds = self.total_frames / self.fps if self.fps else 0
        self.results["match_info"] = {
            "video_path": self.video_path,
            "resolution": f"{self.frame_width}x{self.frame_height}",
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}",
            "team_a": self.team_a_name,
            "team_b": self.team_b_name
        }
        
        # Half timestamps
        if self.half_detector:
            self.results["half_timestamps"] = self.half_detector.get_half_timestamps(self.fps)
        
        # Team directions
        if self.team_direction:
            self.results["team_directions"] = self.team_direction.get_directions_json()
        
        # Events
        self.results["passes"] = pass_events
        self.results["shots"] = shot_events
        
        if self.gk_save_detector:
            self.results["saves"] = self.gk_save_detector.get_save_events()
        
        # Compile statistics
        self.results["statistics"] = self._compile_statistics(pass_events, shot_events)
        
        return self.results
    
    def _compile_statistics(self, pass_events: List, shot_events: List) -> Dict:
        """Compile detailed match statistics split by halves."""
        
        halves = self.results.get("half_timestamps", {})
        h1_start_f = halves.get("h1_start", {}).get("frame")
        h1_end_f = halves.get("h1_end", {}).get("frame")
        h2_start_f = halves.get("h2_start", {}).get("frame")
        h2_end_f = halves.get("h2_end", {}).get("frame")
        
        def is_in_half(frame, half="h1"):
            if half == "h1":
                if h1_start_f is None: return False
                return h1_start_f <= frame <= (h1_end_f if h1_end_f else float('inf'))
            else:
                if h2_start_f is None: return False
                return h2_start_f <= frame <= (h2_end_f if h2_end_f else float('inf'))

        def create_team_stats():
            return {
                "goals": 0,
                "shots_on_target": 0,
                "shots_off_target": 0,
                "short_passes": 0,
                "long_balls": 0,
                "total_passes": 0,
                "attacking_direction": "Unknown"
            }

        stats = {
            "full_match": {
                self.team_a_name: create_team_stats(),
                self.team_b_name: create_team_stats()
            },
            "first_half": {
                "team_a": create_team_stats(),
                "team_b": create_team_stats(),
                "context": ""
            },
            "second_half": {
                "team_a": create_team_stats(),
                "team_b": create_team_stats(),
                "context": ""
            }
        }

        # Helper to categorize passes
        SHORT_PASS_LIMIT = 200 # pixels (heuristic)

        # Process Passes
        for p in pass_events:
            frame = p.get("frame", 0)
            team = p.get("from_team", "Unknown")
            dist = p.get("distance_px", 0)
            
            target_halves = ["full_match"]
            if is_in_half(frame, "h1"): target_halves.append("first_half")
            elif is_in_half(frame, "h2"): target_halves.append("second_half")
            
            for h in target_halves:
                t_key = team if h == "full_match" else ("team_a" if team == self.team_a_name else "team_b")
                if t_key in stats[h]:
                    stats[h][t_key]["total_passes"] += 1
                    if dist > SHORT_PASS_LIMIT:
                        stats[h][t_key]["long_balls"] += 1
                    else:
                        stats[h][t_key]["short_passes"] += 1

        # Process Shots
        for s in shot_events:
            frame = s.get("frame", 0)
            team = s.get("team", "Unknown")
            s_type = s.get("shot_type", "Unknown")
            
            target_halves = ["full_match"]
            if is_in_half(frame, "h1"): target_halves.append("first_half")
            elif is_in_half(frame, "h2"): target_halves.append("second_half")
            
            for h in target_halves:
                t_key = team if h == "full_match" else ("team_a" if team == self.team_a_name else "team_b")
                if t_key in stats[h]:
                    if s_type == "Goal":
                        stats[h][t_key]["goals"] += 1
                        stats[h][t_key]["shots_on_target"] += 1
                    elif s_type == "Shot on target":
                        stats[h][t_key]["shots_on_target"] += 1
                    elif s_type == "Shot off target":
                        stats[h][t_key]["shots_off_target"] += 1

        # Add Attacking Directions
        directions = self.results.get("team_directions", {})
        for half_key, h_stat in [("first_half", stats["first_half"]), ("second_half", stats["second_half"])]:
            h_dir = directions.get(half_key, {})
            t1_data = h_dir.get("team_1", {})
            t2_data = h_dir.get("team_2", {})
            
            if t1_data.get("team_color") == self.team_a_name:
                h_stat["team_a"]["attacking_direction"] = t1_data.get("attacking_direction", "Unknown")
                h_stat["team_b"]["attacking_direction"] = t2_data.get("attacking_direction", "Unknown")
            else:
                h_stat["team_a"]["attacking_direction"] = t2_data.get("attacking_direction", "Unknown")
                h_stat["team_b"]["attacking_direction"] = t1_data.get("attacking_direction", "Unknown")

            # Generate Insight Context
            dir_a = h_stat["team_a"]["attacking_direction"]
            dir_b = h_stat["team_b"]["attacking_direction"]
            arrow_a = "➡" if "left_to_right" in str(dir_a).lower() else "⬅" if "right_to_left" in str(dir_a).lower() else "?"
            arrow_b = "➡" if "left_to_right" in str(dir_b).lower() else "⬅" if "right_to_left" in str(dir_b).lower() else "?"
            
            h_stat["context"] = f"{self.team_a_name} attacking {arrow_a} | {self.team_b_name} attacking {arrow_b}"

        # Calculate Possession Percentages
        for h_key in ["full_match", "first_half", "second_half"]:
            h_data = stats[h_key]
            
            # Map keys based on half type
            if h_key == "full_match":
                t1_key, t2_key = self.team_a_name, self.team_b_name
            else:
                t1_key, t2_key = "team_a", "team_b"
                
            p1 = h_data.get(t1_key, {}).get("total_passes", 0)
            p2 = h_data.get(t2_key, {}).get("total_passes", 0)
            total = p1 + p2
            
            if total > 0:
                h_data[t1_key]["possession"] = round((p1 / total) * 100)
                h_data[t2_key]["possession"] = 100 - h_data[t1_key]["possession"]
            else:
                h_data[t1_key]["possession"] = 50
                h_data[t2_key]["possession"] = 50

        return stats
    
    def save_results(self, output_path: str = "mvp_match_analysis.json") -> str:
        """
        Save complete analysis results to JSON file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ Complete MVP analysis saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary of all analysis results."""
        print("\n" + "=" * 80)
        print("📊 SCOUTME MVP ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Match info
        info = self.results.get("match_info", {})
        print(f"\n📹 Video: {info.get('video_path', 'Unknown')}")
        print(f"   Duration: {info.get('duration', 'Unknown')}")
        print(f"   Resolution: {info.get('resolution', 'Unknown')} @ {info.get('fps', 'Unknown')} fps")
        
        # Half timestamps
        halves = self.results.get("half_timestamps", {})
        print(f"\n⏱️  HALF TIMESTAMPS:")
        print(f"   1st Half: {halves.get('h1_start', {}).get('time', 'Not detected')} → {halves.get('h1_end', {}).get('time', 'Not detected')}")
        print(f"   2nd Half: {halves.get('h2_start', {}).get('time', 'Not detected')} → {halves.get('h2_end', {}).get('time', 'Not detected')}")
        
        # Team directions
        directions = self.results.get("team_directions", {})
        if directions:
            print(f"\n🎯 TEAM DIRECTIONS:")
            for half in ["first_half", "second_half"]:
                half_data = directions.get(half, {})
                print(f"   {half.replace('_', ' ').title()}:")
                for team_key in ["team_1", "team_2"]:
                    team_data = half_data.get(team_key, {})
                    if team_data:
                        print(f"      {team_data.get('team_color', 'Unknown')}: Defends {team_data.get('defending_goal', '?')}, Attacks {team_data.get('attacking_direction', '?')}")
        
        # Statistics
        stats = self.results.get("statistics", {})
        print(f"\n📈 STATISTICS:")
        print(f"   Total Passes: {stats.get('total_passes', 0)}")
        print(f"   Total Shots: {stats.get('total_shots', 0)}")
        print(f"   Total Saves: {stats.get('total_saves', 0)}")
        
        by_team = stats.get("by_team", {})
        for team_name, team_stats in by_team.items():
            if team_stats:
                pass_success = team_stats.get('successful_passes', 0) / max(1, team_stats.get('passes', 1)) * 100
                print(f"\n   {team_name}:")
                print(f"      Passes: {team_stats.get('passes', 0)} ({pass_success:.1f}% successful)")
                print(f"      Shots: {team_stats.get('shots', 0)} ({team_stats.get('shots_on_target', 0)} on target)")
                print(f"      Goals: {team_stats.get('goals', 0)}")
        
        print("\n" + "=" * 80)


# === HELPER FUNCTION FOR EASY INTEGRATION ===

def integrate_with_main4(mvp_analyzer: MVPAnalyzer, frame_idx: int, frame,
                         player_positions: Dict, player_teams: Dict,
                         player_bboxes: Dict, ball_xy: Optional[Tuple],
                         prev_positions: Dict = None,
                         recent_shot_event: Dict = None) -> Dict:
    """
    Helper function to integrate MVP modules with main4.py processing loop.
    
    Call this function in the main processing loop AFTER player/ball detection.
    
    Args:
        mvp_analyzer: Initialized MVPAnalyzer instance
        frame_idx: Current frame index
        frame: Current video frame
        player_positions: Dictionary of player positions {tid: (x, y)}
        player_teams: Dictionary of player teams {tid: team_name}
        player_bboxes: Dictionary of player bboxes {tid: (x1, y1, x2, y2)}
        ball_xy: Ball position or None
        prev_positions: Previous frame player positions (optional)
        recent_shot_event: Most recent shot event (for save detection)
    
    Returns:
        Dictionary with all module outputs for this frame
    """
    outputs = mvp_analyzer.process_frame(
        frame_idx, frame, player_positions, player_teams,
        player_bboxes, ball_xy, prev_positions
    )
    
    # Check for save if there was a recent shot
    if recent_shot_event:
        save = mvp_analyzer.check_for_save(
            frame, frame_idx, recent_shot_event,
            player_positions, player_teams, player_bboxes,
            ball_xy
        )
        if save:
            outputs["save_detected"] = save
    
    return outputs


# === CONFIGURATION TEMPLATE ===

DEFAULT_CONFIG = {
    # Half detection
    "half_detection": {
        "enabled": True,
        "playing_threshold": 0.65,
        "break_threshold": 0.30,
        "min_half_duration_minutes": 40,
        "max_half_duration_minutes": 50
    },
    
    # Team direction detection
    "team_direction": {
        "enabled": True,
        "auto_swap_second_half": True
    },
    
    # GK save detection
    "gk_save_detection": {
        "enabled": True,
        "detection_window_seconds": 2.0
    },
    
    # Video enhancement
    "video_enhancement": {
        "enabled": False,  # Set to True to enable pre-processing
        "enhancement_level": "auto"  # "auto", "light", "medium", "heavy"
    }
}
