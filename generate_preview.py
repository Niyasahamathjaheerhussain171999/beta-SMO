import sys
import os

# Add current directory to path so we can import the generator
sys.path.append(os.getcwd())

try:
    from scout_report_generator import generate_full_scout_report_html
except ImportError:
    print("Error: scout_report_generator.py not found in current directory.")
    sys.exit(1)

# High-fidelity sample data for the preview
sample_passes = [
    {"time": "0:02", "frame": 50, "from_player": 10, "to_player": 8, "from_team": "Blue", "to_team": "Blue", "pass_type": "Short pass", "result": "Success", "distance_px": 85.5, "confidence": 98},
    {"time": "0:15", "frame": 375, "from_player": 5, "to_player": 11, "from_team": "Red", "to_team": "Red", "pass_type": "Long pass", "result": "Success", "distance_px": 420.2, "confidence": 94},
    {"time": "0:32", "frame": 800, "from_player": 11, "to_player": 2, "from_team": "Red", "to_team": "Blue", "pass_type": "Long pass", "result": "Fail", "distance_px": 350.5, "confidence": 91}
]

sample_shots = [
    {"time": "1:52", "frame": 2800, "shooter_id": 10, "shooter_team": "Blue", "shot_type": "Goal", "confidence": 99}
]

# MVP Results with "Tale of Two Halves" data
mvp_results = {
    "half_timestamps": {
        "h1_start": {"time": "00:00", "frame": 0},
        "h1_end": {"time": "45:00", "frame": 67500},
        "h2_start": {"time": "45:00", "frame": 67501},
        "h2_end": {"time": "90:00", "frame": 135000}
    },
    "statistics": {
        "full_match": {
            "Blue": {"goals": 2, "total_passes": 420, "possession": 62},
            "White": {"goals": 1, "total_passes": 310, "possession": 38}
        },
        "first_half": {
            "team_a": {"goals": 0, "shots_on_target": 2, "shots_off_target": 3, "short_passes": 150, "long_balls": 20, "total_passes": 170, "possession": 58, "attacking_direction": "left_to_right"},
            "team_b": {"goals": 1, "shots_on_target": 2, "shots_off_target": 1, "short_passes": 90, "long_balls": 35, "total_passes": 125, "possession": 42, "attacking_direction": "right_to_left"},
            "context": "Blue attacking ➡ | White attacking ⬅"
        },
        "second_half": {
            "team_a": {"goals": 2, "shots_on_target": 3, "shots_off_target": 1, "short_passes": 220, "long_balls": 30, "total_passes": 250, "possession": 68, "attacking_direction": "right_to_left"},
            "team_b": {"goals": 0, "shots_on_target": 1, "shots_off_target": 1, "short_passes": 100, "long_balls": 85, "total_passes": 185, "possession": 32, "attacking_direction": "left_to_right"},
            "context": "Blue attacking ⬅ | White attacking ➡"
        }
    }
}

# Video paths
video_path = "../veo_match_video.mp4" 
annotated_video_path = "../veo_match_video.mp4"

# Generate the report
report_path = generate_full_scout_report_html(
    sample_passes, 
    sample_shots, 
    video_path,
    annotated_video_path,
    {}, {}, 
    25.0,
    mvp_results=mvp_results
)

# Move the report to the preview folder
preview_filename = os.path.join("scout_preview", "scout_match_report_preview.html")
if os.path.exists(preview_filename):
    os.remove(preview_filename)
os.rename(report_path, preview_filename)

print(f"Preview report generated successfully: {preview_filename}")
