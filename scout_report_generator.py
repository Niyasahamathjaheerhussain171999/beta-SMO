"""
ScoutMe - Enhanced Scout Report HTML Generator
Generates interactive HTML report with:
- Pass events table
- Shot on goal events table  
- Popup video player using ANNOTATED video
- Clickable timestamps for instant playback

Author: ScoutMe AI
Version: 6.0
"""

import os
from datetime import datetime


# === COLOR SCHEMES ===
PASS_COLORS = {
    "Short pass": "#4CAF50",
    "Long pass": "#2196F3",
    "Cross": "#FF9800",
    "Short throw-in": "#9C27B0",
    "Long throw-in": "#673AB7",
    "Header": "#F44336"
}

SHOT_COLORS = {
    "Shot on target": "#22c55e",
    "Shot off target": "#ef4444",
    "Shot blocked": "#f59e0b",
    "Goal": "#ffd700"
}


def generate_full_scout_report_html(pass_events, shot_events, video_path, 
                                     annotated_video_path, pass_stats, shot_stats, fps, mvp_results=None):
    """
    Generate comprehensive HTML Scout Match Report with:
    - Pass Analysis
    - Shot on Goal Analysis
    - Video playback with ANNOTATED video in popup
    """
    
    video_name = os.path.basename(video_path)
    annotated_video_name = os.path.basename(annotated_video_path) if annotated_video_path else video_name
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate totals
    total_passes = len(pass_events)
    total_shots = len(shot_events)
    
    # Team Names and Data Extraction
    h_stats = mvp_results.get("statistics", {}) if mvp_results else {}
    full_match_stats = h_stats.get("full_match", {})
    
    # Try to find team names from stats if not directly available
    team_names = list(full_match_stats.keys())
    team_a_name = team_names[0] if len(team_names) > 0 else "Team A"
    team_b_name = team_names[1] if len(team_names) > 1 else "Team B"
    
    # Stats for Overview
    t1_stats = full_match_stats.get(team_a_name, {})
    t2_stats = full_match_stats.get(team_b_name, {})
    
    team_a_goals = t1_stats.get("goals", sum(1 for s in shot_events if s.get('team') == team_a_name and s.get('shot_type') == 'Goal'))
    team_b_goals = t2_stats.get("goals", sum(1 for s in shot_events if s.get('team') == team_b_name and s.get('shot_type') == 'Goal'))
    
    team_a_possession = t1_stats.get("possession", 50)
    team_b_possession = t2_stats.get("possession", 50)
    
    team_a_passes = sum(1 for p in pass_events if p.get('from_team') == team_a_name)
    team_b_passes = sum(1 for p in pass_events if p.get('from_team') == team_b_name)
    
    # Pass type counts
    pass_type_counts = {}
    for p in pass_events:
        pt = p.get('pass_type', 'Unknown')
        pass_type_counts[pt] = pass_type_counts.get(pt, 0) + 1
    
    # Shot type counts
    shot_type_counts = {}
    for s in shot_events:
        st = s.get('shot_type', 'Unknown')
        shot_type_counts[st] = shot_type_counts.get(st, 0) + 1
    
    # Generate timeline markers for JavaScript
    markers_js = generate_markers_js(pass_events, shot_events, fps)
    
    # Extract Half Stats
    h1 = h_stats.get("first_half", {"team_a": {}, "team_b": {}, "context": "No data available"})
    h2 = h_stats.get("second_half", {"team_a": {}, "team_b": {}, "context": "No data available"})
    timestamps = mvp_results.get("half_timestamps", {}) if mvp_results else {}
    h1_range = f"{timestamps.get('h1_start', {}).get('time', '00:00')} – {timestamps.get('h1_end', {}).get('time', '45:00')}"
    h2_range = f"{timestamps.get('h2_start', {}).get('time', '45:00')} – {timestamps.get('h2_end', {}).get('time', '90:00')}"
    
    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScoutMe - Complete Match Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1117 100%);
            min-height: 100vh;
            color: #e2e8f0;
        }}
        
        .header {{
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            padding: 25px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        }}
        
        .logo {{
            font-size: 32px;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 12px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .logo span {{
            background: linear-gradient(135deg, #ffffff, #e2e8f0);
            color: #6366f1;
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 24px;
        }}
        
        .header-info {{
            text-align: right;
            font-size: 14px;
            opacity: 0.95;
        }}
        
        .container {{
            max-width: 1700px;
            margin: 0 auto;
            padding: 30px;
        }}
        
        /* === TAB NAVIGATION === */
        .tab-nav {{
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            background: rgba(255,255,255,0.05);
            padding: 8px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}
        
        .tab-btn {{
            flex: 1;
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            background: transparent;
            color: #94a3b8;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}
        
        .tab-btn:hover {{
            background: rgba(255,255,255,0.1);
            color: #ffffff;
        }}
        
        .tab-btn.active {{
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.3s ease;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* === VIDEO SECTION === */
        .video-section {{
            background: rgba(30, 30, 60, 0.6);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .video-title {{
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #ffffff;
        }}
        
        .video-container {{
            position: relative;
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        #matchVideo {{
            width: 100%;
            border-radius: 15px;
            background: #000;
        }}
        
        .video-error {{
            display: none;
            padding: 30px;
            background: rgba(239, 68, 68, 0.2);
            border-radius: 15px;
            text-align: center;
            border: 2px dashed rgba(239, 68, 68, 0.5);
        }}
        
        /* === VIDEO POPUP MODAL (ANNOTATED VIDEO) === */
        .video-popup-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.97);
            z-index: 10000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}
        
        .video-popup-overlay.active {{
            display: flex;
        }}
        
        .video-popup-container {{
            width: 92%;
            max-width: 1300px;
            position: relative;
        }}
        
        .video-popup-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 25px;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            border-radius: 18px 18px 0 0;
        }}
        
        .video-popup-title {{
            font-size: 20px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .video-popup-badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }}
        
        .video-popup-close {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            font-size: 28px;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }}
        
        .video-popup-close:hover {{
            background: rgba(255,255,255,0.4);
            transform: scale(1.1);
        }}
        
        .video-popup-player {{
            width: 100%;
            background: #000;
        }}
        
        #popupVideo {{
            width: 100%;
            max-height: 70vh;
        }}
        
        .video-popup-info {{
            margin-top: 18px;
            padding: 22px;
            background: rgba(255,255,255,0.08);
            border-radius: 15px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .popup-info-item {{
            text-align: center;
        }}
        
        .popup-info-label {{
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .popup-info-value {{
            font-size: 18px;
            font-weight: 700;
        }}
        
        .popup-annotated-badge {{
            background: linear-gradient(90deg, #22c55e, #16a34a);
            color: white;
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 15px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.8; }}
        }}
        
        /* === VIDEO CONTROLS === */
        .video-controls {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 18px;
            padding: 18px;
            background: rgba(0,0,0,0.4);
            border-radius: 12px;
        }}
        
        .play-btn {{
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            border: none;
            color: white;
            width: 55px;
            height: 55px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .play-btn:hover {{
            transform: scale(1.1);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        }}
        
        .timeline-container {{
            flex: 1;
            position: relative;
        }}
        
        .timeline {{
            width: 100%;
            height: 14px;
            background: rgba(255,255,255,0.15);
            border-radius: 7px;
            cursor: pointer;
            position: relative;
            overflow: visible;
        }}
        
        .timeline-progress {{
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            border-radius: 7px;
            width: 0%;
            transition: width 0.1s linear;
        }}
        
        .timeline-marker {{
            position: absolute;
            top: -10px;
            width: 5px;
            height: 34px;
            border-radius: 3px;
            cursor: pointer;
            transition: transform 0.2s;
            z-index: 10;
        }}
        
        .timeline-marker:hover {{
            transform: scaleY(1.4) scaleX(1.3);
        }}
        
        .timeline-marker:hover::after {{
            content: attr(data-info);
            position: absolute;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.95);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 100;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .time-display {{
            font-family: 'Courier New', monospace;
            font-size: 15px;
            min-width: 110px;
            text-align: center;
            font-weight: 600;
        }}
        
        /* === SUMMARY CARDS === */
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 18px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 22px 18px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }}
        
        .card-value {{
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 8px;
        }}
        
        .card-label {{
            font-size: 12px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 500;
        }}
        
        .card.total .card-value {{ color: #fbbf24; }}
        .card.blue .card-value {{ color: #3b82f6; }}
        .card.red .card-value {{ color: #ef4444; }}
        .card.success .card-value {{ color: #22c55e; }}
        .card.shots .card-value {{ color: #a855f7; }}
        .card.goals .card-value {{ color: #ffd700; text-shadow: 0 0 20px rgba(255,215,0,0.5); }}
        
        /* === BADGE FILTERS === */
        .badge-filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 25px;
            justify-content: center;
        }}
        
        .filter-badge {{
            padding: 12px 22px;
            border-radius: 30px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            border: none;
            color: white;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .filter-badge:hover {{
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        
        .filter-badge .count {{
            background: rgba(255,255,255,0.25);
            padding: 3px 12px;
            border-radius: 15px;
            font-size: 12px;
        }}
        
        /* === DATA TABLE === */
        .table-container {{
            background: rgba(30, 30, 60, 0.5);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
        }}
        
        .table-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .table-title {{
            font-size: 20px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .search-box {{
            padding: 12px 20px;
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.08);
            color: white;
            font-size: 14px;
            width: 250px;
            transition: all 0.3s;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #8b5cf6;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        }}
        
        .filter-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            padding: 10px 20px;
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            background: transparent;
            color: #94a3b8;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .filter-btn:hover, .filter-btn.active {{
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            border-color: transparent;
            color: white;
        }}
        
        .table-scroll {{
            overflow-x: auto;
            max-height: 600px;
            overflow-y: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: rgba(99, 102, 241, 0.3);
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: sticky;
            top: 0;
            z-index: 5;
        }}
        
        td {{
            padding: 14px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        
        tbody tr {{
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        tbody tr:hover {{
            background: rgba(99, 102, 241, 0.15);
            transform: scale(1.005);
        }}
        
        tbody tr.active-row {{
            background: rgba(139, 92, 246, 0.25);
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        }}
        
        .timestamp-btn {{
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        .timestamp-btn:hover {{
            transform: scale(1.08);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.5);
        }}
        
        .player-cell {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .player-avatar {{
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
            color: white;
        }}
        
        .player-avatar.blue {{ background: linear-gradient(135deg, #3b82f6, #1d4ed8); }}
        .player-avatar.red {{ background: linear-gradient(135deg, #ef4444, #b91c1c); }}
        .player-avatar.unknown {{ background: linear-gradient(135deg, #6b7280, #4b5563); }}
        
        .type-tag {{
            padding: 6px 14px;
            border-radius: 18px;
            font-size: 12px;
            font-weight: 600;
            color: white;
        }}
        
        .result-badge {{
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 11px;
            font-weight: 600;
        }}
        
        .result-badge.success {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }}
        .result-badge.fail {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .result-badge.unknown {{ background: rgba(107, 114, 128, 0.2); color: #9ca3af; border: 1px solid rgba(107, 114, 128, 0.3); }}
        
        .confidence-bar {{
            width: 60px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        
        /* === FOOTER === */
        .footer {{
            text-align: center;
            padding: 30px;
            opacity: 0.7;
            font-size: 13px;
        }}
        
        .footer a {{
            color: #8b5cf6;
            text-decoration: none;
        }}

        /* === TALE OF TWO HALVES === */
        .halves-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .half-card {{
            background: linear-gradient(rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.96)), 
                        radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 80%);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.4);
            position: relative;
            overflow: hidden;
        }}
        
        .half-card::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: url("data:image/svg+xml,%3Csvg width='400' height='200' viewBox='0 0 400 200' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M200 0V200M0 100H400M200 100C200 111.046 191.046 120 180 120C168.954 120 160 111.046 160 100C160 88.9543 168.954 80 180 80C191.046 80 200 88.9543 200 100Z' stroke='rgba(255,255,255,0.03)' stroke-width='2'/%3E%3C/svg%3E");
            background-size: cover;
            opacity: 0.5;
            pointer-events: none;
        }}
        
        .half-header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 25px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        
        .half-title {{
            font-size: 28px;
            font-weight: 900;
            color: #fbbf24;
            letter-spacing: 2.5px;
            text-transform: uppercase;
        }}
        
        .half-context {{
            font-size: 14px;
            color: #94a3b8;
            margin-top: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }}

        .orientation-row {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
        }}

        .orientation-badge {{
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 8px;
            background: #1e293b;
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .orientation-badge.attack {{ color: #10b981; }} /* Green */
        .orientation-badge.defend {{ color: #f43f5e; }} /* Rose/Red */
        
        .team-indicator {{
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #f8fafc;
        }}
        
        .comparison-row {{
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }}
        
        .comparison-row:last-child {{ border: none; }}
        
        .comp-val {{
            font-size: 26px;
            font-weight: 900;
            text-align: center;
        }}
        
        .comp-val.team-a {{ color: #3b82f6; text-shadow: 0 0 10px rgba(59, 130, 246, 0.2); }}
        .comp-val.team-b {{ color: #ef4444; text-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }}
        
        .comp-metric {{
            text-align: center;
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 2.5px;
            color: #64748b;
        }}
        
        .half_insight {{
            margin-top: 25px;
            padding: 20px;
            background: rgba(99, 102, 241, 0.1);
            border-left: 4px solid #6366f1;
            border-radius: 0 12px 12px 0;
            font-style: italic;
            font-size: 15px;
            line-height: 1.6;
            color: #cbd5e1;
        }}

        /* === TV MATCH SUMMARY CARD === */
        .tv-card-container {{
            margin-bottom: 40px;
            display: flex;
            justify-content: center;
        }}

        .tv-card {{
            background: #0f172a;
            width: 100%;
            max-width: 800px;
            border-radius: 12px;
            border: 2px solid #1e293b;
            overflow: hidden;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            font-family: 'Inter', system-ui, sans-serif;
        }}

        .tv-card-header {{
            background: #1e293b;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid #334155;
        }}

        .tv-main-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 30px 40px;
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        }}

        .tv-team-name {{
            font-size: 24px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 40%;
        }}

        .tv-team-name.left {{ text-align: right; color: #3b82f6; }}
        .tv-team-name.right {{ text-align: left; color: #ef4444; }}

        .tv-score-box {{
            font-size: 48px;
            font-weight: 900;
            color: white;
            display: flex;
            align-items: center;
            gap: 20px;
            text-shadow: 0 0 20px rgba(255,255,255,0.2);
        }}

        .tv-score-divider {{
            opacity: 0.5;
            font-weight: 300;
        }}

        .tv-dominance-section {{
            background: #0f172a;
            padding: 20px 40px 30px;
            text-align: center;
        }}

        .tv-dominance-title {{
            font-size: 12px;
            font-weight: 800;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 15px;
        }}

        .tv-dominance-bar-container {{
            height: 35px;
            background: #1e293b;
            border-radius: 4px;
            display: flex;
            overflow: hidden;
            position: relative;
        }}

        .tv-bar {{
            height: 100%;
            display: flex;
            align-items: center;
            padding: 0 15px;
            font-weight: 800;
            font-size: 14px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .tv-bar.team-a {{
            background: #2563eb;
            color: white;
            justify-content: flex-start;
        }}

        .tv-bar.team-b {{
            background: #dc2626;
            color: white;
            justify-content: flex-end;
        }}

        .tv-possession-label {{
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
    </style>
</head>
<body>
    <!-- HEADER -->
    <header class="header">
        <div class="logo">
            ⚽ <span>ScoutMe</span> AI Match Analysis
        </div>
        <div class="header-info">
            <div style="font-weight: 600; font-size: 16px;">📊 Complete Match Report</div>
            <div>Generated: {report_time}</div>
            <div>Video: {video_name}</div>
        </div>
    </header>
    
    <div class="container">
        <!-- TAB NAVIGATION -->
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('overview')">
                📊 Overview
            </button>
            <button class="tab-btn" onclick="switchTab('passes')">
                🎯 Passes ({total_passes})
            </button>
            <button class="tab-btn" onclick="switchTab('shots')">
                ⚽ Shots ({total_shots})
            </button>
            <button class="tab-btn" onclick="switchTab('halves')">
                ⏱️ Halves
            </button>
            <button class="tab-btn" onclick="switchTab('video')">
                🎬 Video
            </button>
        </div>

        <!-- OVERVIEW TAB -->
        <div class="tab-content active" id="overview-tab">
            <div class="tv-card-container">
                <div class="tv-card">
                    <div class="tv-card-header">1. Match Summary Card (TV Standard)</div>
                    <div class="tv-main-row">
                        <div class="tv-team-name left">TEAM {team_a_name.upper()}</div>
                        <div class="tv-score-box">
                            <span>{team_a_goals}</span>
                            <span class="tv-score-divider">-</span>
                            <span>{team_b_goals}</span>
                        </div>
                        <div class="tv-team-name right">TEAM {team_b_name.upper()}</div>
                    </div>
                    <div class="tv-dominance-section">
                        <div class="tv-dominance-title">DOMINANCE BAR</div>
                        <div class="tv-dominance-bar-container">
                            <div class="tv-bar team-a" style="width: {team_a_possession}%">
                                <span class="tv-possession-label">{team_a_possession}% POSSESSION</span>
                            </div>
                            <div class="tv-bar team-b" style="width: {team_b_possession}%">
                                <span class="tv-possession-label">{team_b_possession}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Global Stats Below TV Card -->
            <div class="summary-cards" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="card total">
                    <div class="card-value">{total_passes}</div>
                    <div class="card-label">Total Passes</div>
                </div>
                <div class="card success">
                    <div class="card-value">{sum(1 for p in pass_events if p.get('result') == 'Success')}</div>
                    <div class="card-label">Pass Success</div>
                </div>
                <div class="card blue">
                    <div class="card-value">{team_a_passes}</div>
                    <div class="card-label">{team_a_name} Passes</div>
                </div>
                <div class="card red">
                    <div class="card-value">{team_b_passes}</div>
                    <div class="card-label">{team_b_name} Passes</div>
                </div>
            </div>
        </div>
        
        <!-- PASS ANALYSIS TAB -->
        <div class="tab-content" id="passes-tab">
            <div class="summary-cards">
                <div class="card total">
                    <div class="card-value">{total_passes}</div>
                    <div class="card-label">Total Passes</div>
                </div>
                <div class="card blue">
                    <div class="card-value">{team_a_passes}</div>
                    <div class="card-label">{team_a_name}</div>
                </div>
                <div class="card red">
                    <div class="card-value">{team_b_passes}</div>
                    <div class="card-label">{team_b_name}</div>
                </div>
                <div class="card success">
                    <div class="card-value">{sum(1 for p in pass_events if p.get('result') == 'Success')}</div>
                    <div class="card-label">Successful</div>
                </div>
                <div class="card">
                    <div class="card-value" style="color: #f97316;">{sum(1 for p in pass_events if p.get('result') == 'Fail')}</div>
                    <div class="card-label">Failed</div>
                </div>
            </div>
            
            <div class="badge-filters">
'''
    
    # Add pass type filter badges
    for pass_type, color in PASS_COLORS.items():
        count = pass_type_counts.get(pass_type, 0)
        html_content += f'''                <button class="filter-badge" style="background: {color};" onclick="filterByType('pass', '{pass_type}')">
                    {pass_type} <span class="count">{count}</span>
                </button>
'''
    
    html_content += '''            </div>
            
            <div class="table-container">
                <div class="table-header">
                    <div class="table-title">📋 Pass Events - Click to watch with AI annotations!</div>
                    <input type="text" class="search-box" placeholder="🔍 Search passes..." onkeyup="filterTable('passTable', this.value)">
                </div>
                
                <div class="filter-buttons">
                    <button class="filter-btn active" onclick="filterByTeam('passTable', 'all')">All</button>
                    <button class="filter-btn" onclick="filterByTeam('passTable', 'Blue')">🔵 Blue</button>
                    <button class="filter-btn" onclick="filterByTeam('passTable', 'Red')">🔴 Red</button>
                    <button class="filter-btn" onclick="filterByResult('passTable', 'Success')">✅ Success</button>
                    <button class="filter-btn" onclick="filterByResult('passTable', 'Fail')">❌ Failed</button>
                </div>
                
                <div class="table-scroll">
                    <table id="passTable">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>▶ Watch</th>
                                <th>From</th>
                                <th>To</th>
                                <th>Type</th>
                                <th>Result</th>
                                <th>Conf</th>
                            </tr>
                        </thead>
                        <tbody>
'''
    
    # Add pass table rows
    for idx, event in enumerate(pass_events, 1):
        html_content += generate_pass_row(idx, event, fps)
    
    html_content += '''                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- SHOT ANALYSIS TAB -->
        <div class="tab-content" id="shots-tab">
            <div class="summary-cards">
                <div class="card shots">
                    <div class="card-value">''' + str(total_shots) + '''</div>
                    <div class="card-label">Total Shots</div>
                </div>
                <div class="card">
                    <div class="card-value" style="color: #22c55e;">''' + str(shot_type_counts.get('Shot on target', 0)) + '''</div>
                    <div class="card-label">On Target</div>
                </div>
                <div class="card">
                    <div class="card-value" style="color: #ef4444;">''' + str(shot_type_counts.get('Shot off target', 0)) + '''</div>
                    <div class="card-label">Off Target</div>
                </div>
                <div class="card">
                    <div class="card-value" style="color: #f59e0b;">''' + str(shot_type_counts.get('Shot blocked', 0)) + '''</div>
                    <div class="card-label">Blocked</div>
                </div>
                <div class="card goals">
                    <div class="card-value">''' + str(shot_type_counts.get('Goal', 0)) + '''</div>
                    <div class="card-label">⚽ Goals!</div>
                </div>
            </div>
            
            <div class="badge-filters">
'''
    
    # Add shot type filter badges
    for shot_type, color in SHOT_COLORS.items():
        count = shot_type_counts.get(shot_type, 0)
        html_content += f'''                <button class="filter-badge" style="background: {color};" onclick="filterByType('shot', '{shot_type}')">
                    {shot_type} <span class="count">{count}</span>
                </button>
'''
    
    html_content += '''            </div>
            
            <div class="table-container">
                <div class="table-header">
                    <div class="table-title">🎯 Shot Events - Click to watch with AI annotations!</div>
                    <input type="text" class="search-box" placeholder="🔍 Search shots..." onkeyup="filterTable('shotTable', this.value)">
                </div>
                
                <div class="filter-buttons">
                    <button class="filter-btn active" onclick="filterByTeam('shotTable', 'all')">All</button>
                    <button class="filter-btn" onclick="filterByTeam('shotTable', 'Blue')">🔵 Blue</button>
                    <button class="filter-btn" onclick="filterByTeam('shotTable', 'Red')">🔴 Red</button>
                </div>
                
                <div class="table-scroll">
                    <table id="shotTable">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>▶ Watch</th>
                                <th>Shooter</th>
                                <th>Team</th>
                                <th>Shot Type</th>
                                <th>Conf</th>
                            </tr>
                        </thead>
                        <tbody>
'''
    
    # Add shot table rows
    for idx, event in enumerate(shot_events, 1):
        html_content += generate_shot_row(idx, event, fps)
    
    html_content += f'''                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- TALE OF TWO HALVES TAB -->
        <div class="tab-content" id="halves-tab">
            <div class="halves-grid">
                <!-- FIRST HALF -->
                <div class="half-card">
                    <div class="half-header">
                        <div class="half-title">First Half</div>
                        <div class="half-context">{h1_range}</div>
                        
                        <div style="margin-top:20px;">
                            <div class="team-indicator" style="color:#3b82f6;">{team_a_name.upper()}</div>
                            <div class="orientation-row">
                                <div class="orientation-badge attack" style="border-color: rgba(59, 130, 246, 0.4);">
                                    <span>⚔️ Attacking</span>
                                    <span>{'➡' if 'left_to_right' in str(h1.get('team_a', {}).get('attacking_direction')).lower() else '⬅'}</span>
                                </div>
                                <div class="orientation-badge defend" style="border-color: rgba(59, 130, 246, 0.4);">
                                    <span>🛡️ Defending</span>
                                    <span>{'⬅' if 'left_to_right' in str(h1.get('team_a', {}).get('attacking_direction')).lower() else '➡'}</span>
                                </div>
                            </div>
                        </div>

                        <div style="margin-top:20px;">
                            <div class="team-indicator" style="color:#ef4444;">{team_b_name.upper()}</div>
                            <div class="orientation-row">
                                <div class="orientation-badge attack" style="border-color: rgba(239, 68, 68, 0.4);">
                                    <span>⚔️ Attacking</span>
                                    <span>{'➡' if 'left_to_right' in str(h1.get('team_b', {}).get('attacking_direction')).lower() else '⬅'}</span>
                                </div>
                                <div class="orientation-badge defend" style="border-color: rgba(239, 68, 68, 0.4);">
                                    <span>🛡️ Defending</span>
                                    <span>{'⬅' if 'left_to_right' in str(h1.get('team_b', {}).get('attacking_direction')).lower() else '➡'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h1.get('team_a', {}).get('goals', 0)}</div>
                        <div class="comp-metric">Goals</div>
                        <div class="comp-val team-b">{h1.get('team_b', {}).get('goals', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h1.get('team_a', {}).get('shots_on_target', 0)}</div>
                        <div class="comp-metric">Shots on Target</div>
                        <div class="comp-val team-b">{h1.get('team_b', {}).get('shots_on_target', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h1.get('team_a', {}).get('shots_off_target', 0)}</div>
                        <div class="comp-metric">Shots off Target</div>
                        <div class="comp-val team-b">{h1.get('team_b', {}).get('shots_off_target', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h1.get('team_a', {}).get('short_passes', 0)}</div>
                        <div class="comp-metric">Short Passes</div>
                        <div class="comp-val team-b">{h1.get('team_b', {}).get('short_passes', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h1.get('team_a', {}).get('long_balls', 0)}</div>
                        <div class="comp-metric">Long Balls</div>
                        <div class="comp-val team-b">{h1.get('team_b', {}).get('long_balls', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h1.get('team_a', {}).get('total_passes', 0)}</div>
                        <div class="comp-metric">Total Passes</div>
                        <div class="comp-val team-b">{h1.get('team_b', {}).get('total_passes', 0)}</div>
                    </div>
                    
                    <div class="half-insight">
                        "The first half analytics show {team_a_name if h1.get('team_a', {}).get('total_passes', 0) > h1.get('team_b', {}).get('total_passes', 0) else team_b_name} Team focusing on possession, with {h1.get('team_a', {}).get('long_balls', 0) if h1.get('team_a', {}).get('long_balls', 0) > h1.get('team_b', {}).get('long_balls', 0) else h1.get('team_b', {}).get('long_balls', 0)} long balls attempted to stretch the defense."
                    </div>
                </div>

                <!-- SECOND HALF -->
                <div class="half-card">
                    <div class="half-header">
                        <div class="half-title">Second Half</div>
                        <div class="half-context">{h2_range}</div>
                        
                        <div style="margin-top:20px;">
                            <div class="team-indicator" style="color:#3b82f6;">{team_a_name.upper()}</div>
                            <div class="orientation-row">
                                <div class="orientation-badge attack" style="border-color: rgba(59, 130, 246, 0.4);">
                                    <span>⚔️ Attacking</span>
                                    <span>{'➡' if 'left_to_right' in str(h2.get('team_a', {}).get('attacking_direction')).lower() else '⬅'}</span>
                                </div>
                                <div class="orientation-badge defend" style="border-color: rgba(59, 130, 246, 0.4);">
                                    <span>🛡️ Defending</span>
                                    <span>{'⬅' if 'left_to_right' in str(h2.get('team_a', {}).get('attacking_direction')).lower() else '➡'}</span>
                                </div>
                            </div>
                        </div>

                        <div style="margin-top:20px;">
                            <div class="team-indicator" style="color:#ef4444;">{team_b_name.upper()}</div>
                            <div class="orientation-row">
                                <div class="orientation-badge attack" style="border-color: rgba(239, 68, 68, 0.4);">
                                    <span>⚔️ Attacking</span>
                                    <span>{'➡' if 'left_to_right' in str(h2.get('team_b', {}).get('attacking_direction')).lower() else '⬅'}</span>
                                </div>
                                <div class="orientation-badge defend" style="border-color: rgba(239, 68, 68, 0.4);">
                                    <span>🛡️ Defending</span>
                                    <span>{'⬅' if 'left_to_right' in str(h2.get('team_b', {}).get('attacking_direction')).lower() else '➡'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h2.get('team_a', {}).get('goals', 0)}</div>
                        <div class="comp-metric">Goals</div>
                        <div class="comp-val team-b">{h2.get('team_b', {}).get('goals', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h2.get('team_a', {}).get('shots_on_target', 0)}</div>
                        <div class="comp-metric">Shots on Target</div>
                        <div class="comp-val team-b">{h2.get('team_b', {}).get('shots_on_target', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h2.get('team_a', {}).get('shots_off_target', 0)}</div>
                        <div class="comp-metric">Shots off Target</div>
                        <div class="comp-val team-b">{h2.get('team_b', {}).get('shots_off_target', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h2.get('team_a', {}).get('short_passes', 0)}</div>
                        <div class="comp-metric">Short Passes</div>
                        <div class="comp-val team-b">{h2.get('team_b', {}).get('short_passes', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h2.get('team_a', {}).get('long_balls', 0)}</div>
                        <div class="comp-metric">Long Balls</div>
                        <div class="comp-val team-b">{h2.get('team_b', {}).get('long_balls', 0)}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comp-val team-a">{h2.get('team_a', {}).get('total_passes', 0)}</div>
                        <div class="comp-metric">Total Passes</div>
                        <div class="comp-val team-b">{h2.get('team_b', {}).get('total_passes', 0)}</div>
                    </div>
                    
                    <div class="half-insight">
                        "The match intensity shifted in the second half. {team_a_name if h2.get('team_a', {}).get('goals', 0) > h1.get('team_a', {}).get('goals', 0) else team_b_name} Team increased their attacking efficiency, aiming to capitalize on transition moments."
                    </div>
                </div>
            </div>
        </div>
        
        <!-- FULL VIDEO TAB -->
        <div class="tab-content" id="video-tab">
            <div class="video-section">
                <div class="video-title">🎬 Match Video - with Timeline Markers</div>
                <div class="video-container">
                    <video id="matchVideo" preload="metadata">
                        <source src="{video_name}" type="video/mp4">
                        Your browser does not support video playback.
                    </video>
                    
                    <div class="video-error" id="videoError">
                        <h3 style="margin-bottom: 15px;">⚠️ Video Not Found</h3>
                        <p>Place <strong>{video_name}</strong> in the same folder as this HTML file.</p>
                    </div>
                    
                    <div class="video-controls">
                        <button class="play-btn" onclick="togglePlay()">▶</button>
                        <div class="timeline-container">
                            <div class="timeline" id="timeline" onclick="seekVideo(event)">
                                <div class="timeline-progress" id="timelineProgress"></div>
                            </div>
                        </div>
                        <div class="time-display">
                            <span id="currentTime">0:00</span> / <span id="duration">0:00</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- VIDEO POPUP MODAL - PLAYS ANNOTATED VIDEO -->
    <div class="video-popup-overlay" id="videoPopup">
        <div class="video-popup-container">
            <div class="video-popup-header">
                <div class="video-popup-title">
                    🎬 Action Replay
                    <span class="video-popup-badge" id="popupTypeBadge">Pass</span>
                </div>
                <button class="video-popup-close" onclick="closeVideoPopup()">✕</button>
            </div>
            <div class="video-popup-player">
                <video id="popupVideo" controls>
                    <source src="{annotated_video_name}" type="video/mp4">
                </video>
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <span class="popup-annotated-badge">
                    ✨ Playing ANNOTATED Video with AI Highlights
                </span>
            </div>
            <div class="video-popup-info">
                <div class="popup-info-item">
                    <div class="popup-info-label">Event</div>
                    <div class="popup-info-value" id="popupEventType">-</div>
                </div>
                <div class="popup-info-item">
                    <div class="popup-info-label">Player/From</div>
                    <div class="popup-info-value" id="popupFrom">-</div>
                </div>
                <div class="popup-info-item">
                    <div class="popup-info-label">To</div>
                    <div class="popup-info-value" id="popupTo">-</div>
                </div>
                <div class="popup-info-item">
                    <div class="popup-info-label">Result</div>
                    <div class="popup-info-value" id="popupResult">-</div>
                </div>
                <div class="popup-info-item">
                    <div class="popup-info-label">Time</div>
                    <div class="popup-info-value" id="popupTime">-</div>
                </div>
                <div class="popup-info-item">
                    <div class="popup-info-label">Confidence</div>
                    <div class="popup-info-value" id="popupConfidence">-</div>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="footer">
        <p>Generated by <strong>ScoutMe AI v6</strong> - Soccer Analysis Engine</p>
        <p>🎬 Click any event to watch with AI-annotated video highlighting the action!</p>
    </footer>
    
    <script>
        // === CONFIGURATION ===
        const ANNOTATED_VIDEO = "{annotated_video_name}";
        const ORIGINAL_VIDEO = "{video_name}";
        
        // Pass and shot markers for timeline
        {markers_js}
        
        const video = document.getElementById('matchVideo');
        const popupVideo = document.getElementById('popupVideo');
        const timeline = document.getElementById('timeline');
        const timelineProgress = document.getElementById('timelineProgress');
        const playBtn = document.querySelector('.play-btn');
        
        // === TAB SWITCHING ===
        function switchTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }}
        
        // === VIDEO CONTROLS ===
        function togglePlay() {{
            if (video.paused) {{
                video.play();
                playBtn.textContent = '⏸';
            }} else {{
                video.pause();
                playBtn.textContent = '▶';
            }}
        }}
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins + ':' + (secs < 10 ? '0' : '') + secs;
        }}
        
        video.addEventListener('timeupdate', function() {{
            const percent = (video.currentTime / video.duration) * 100;
            timelineProgress.style.width = percent + '%';
            document.getElementById('currentTime').textContent = formatTime(video.currentTime);
        }});
        
        video.addEventListener('loadedmetadata', function() {{
            document.getElementById('duration').textContent = formatTime(video.duration);
            addTimelineMarkers();
        }});
        
        video.addEventListener('error', function() {{
            document.getElementById('videoError').style.display = 'block';
        }});
        
        video.addEventListener('play', () => playBtn.textContent = '⏸');
        video.addEventListener('pause', () => playBtn.textContent = '▶');
        
        function seekVideo(e) {{
            const rect = timeline.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            video.currentTime = percent * video.duration;
        }}
        
        // === ADD MARKERS TO TIMELINE ===
        function addTimelineMarkers() {{
            const duration = video.duration;
            
            allMarkers.forEach(marker => {{
                const percent = (marker.time / duration) * 100;
                const el = document.createElement('div');
                el.className = 'timeline-marker';
                el.style.left = percent + '%';
                el.style.backgroundColor = marker.color;
                el.dataset.info = '#' + marker.idx + ' ' + marker.type + ' (' + marker.team + ')';
                el.onclick = function(e) {{
                    e.stopPropagation();
                    video.currentTime = Math.max(0, marker.time - 1);
                    video.play();
                }};
                timeline.appendChild(el);
            }});
        }}
        
        // === VIDEO POPUP - PLAYS ANNOTATED VIDEO ===
        function openVideoPopup(eventType, seconds, type, team, playerId, toTeam, toId, result, confidence) {{
            const popup = document.getElementById('videoPopup');
            
            // Get the color for the type
            const passColors = {{'Short pass': '#4CAF50', 'Long pass': '#2196F3', 'Cross': '#FF9800', 'Short throw-in': '#9C27B0', 'Long throw-in': '#673AB7', 'Header': '#F44336'}};
            const shotColors = {{'Shot on target': '#22c55e', 'Shot off target': '#ef4444', 'Shot blocked': '#f59e0b', 'Goal': '#ffd700'}};
            const color = eventType === 'pass' ? (passColors[type] || '#6366f1') : (shotColors[type] || '#6366f1');
            
            // Update popup info
            document.getElementById('popupTypeBadge').textContent = type;
            document.getElementById('popupTypeBadge').style.background = color;
            document.getElementById('popupEventType').textContent = type;
            document.getElementById('popupEventType').style.color = color;
            
            const teamColor = team === 'Blue' ? '#3b82f6' : (team === 'Red' ? '#ef4444' : '#9ca3af');
            document.getElementById('popupFrom').innerHTML = '<span style="color:' + teamColor + '">#' + playerId + ' ' + team + '</span>';
            
            if (eventType === 'pass' && toId) {{
                const toColor = toTeam === 'Blue' ? '#3b82f6' : (toTeam === 'Red' ? '#ef4444' : '#9ca3af');
                document.getElementById('popupTo').innerHTML = '<span style="color:' + toColor + '">#' + toId + ' ' + toTeam + '</span>';
            }} else {{
                document.getElementById('popupTo').textContent = '-';
            }}
            
            document.getElementById('popupResult').textContent = result || '-';
            document.getElementById('popupResult').style.color = result === 'Success' ? '#22c55e' : (result === 'Fail' ? '#ef4444' : '#9ca3af');
            document.getElementById('popupTime').textContent = formatTime(seconds);
            document.getElementById('popupConfidence').textContent = confidence + '%';
            
            // *** KEY: Try ANNOTATED video first, fallback to ORIGINAL ***
            popupVideo.onerror = function() {{
                console.log('Annotated video failed, trying original...');
                popupVideo.src = ORIGINAL_VIDEO;
                popupVideo.currentTime = Math.max(0, seconds - 2);
                popupVideo.play();
            }};
            
            popupVideo.src = ANNOTATED_VIDEO;
            popupVideo.currentTime = Math.max(0, seconds - 2); // Start 2 sec before
            
            popup.classList.add('active');
            document.body.style.overflow = 'hidden';
            
            // Auto-play
            popupVideo.play();
        }}
        
        function closeVideoPopup() {{
            const popup = document.getElementById('videoPopup');
            popupVideo.pause();
            popup.classList.remove('active');
            document.body.style.overflow = '';
        }}
        
        // Close on Escape or click outside
        document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeVideoPopup(); }});
        document.getElementById('videoPopup').addEventListener('click', e => {{ if (e.target.id === 'videoPopup') closeVideoPopup(); }});
        
        // === FILTERING ===
        function filterTable(tableId, searchText) {{
            const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
            searchText = searchText.toLowerCase();
            rows.forEach(row => {{
                row.style.display = row.textContent.toLowerCase().includes(searchText) ? '' : 'none';
            }});
        }}
        
        function filterByTeam(tableId, team) {{
            const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
            rows.forEach(row => {{
                if (team === 'all') {{
                    row.style.display = '';
                }} else {{
                    row.style.display = row.dataset.team === team ? '' : 'none';
                }}
            }});
            
            // Update active button
            const container = document.querySelector('#' + tableId).closest('.table-container');
            container.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}
        
        function filterByResult(tableId, result) {{
            const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
            rows.forEach(row => {{
                row.style.display = row.dataset.result === result ? '' : 'none';
            }});
            
            const container = document.querySelector('#' + tableId).closest('.table-container');
            container.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}
        
        function filterByType(eventType, type) {{
            const tableId = eventType === 'pass' ? 'passTable' : 'shotTable';
            const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
            rows.forEach(row => {{
                row.style.display = row.dataset.type === type ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
'''
    
    # Save HTML file
    output_path = "scout_match_report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def generate_pass_row(idx, event, fps):
    """Generate HTML table row for a pass event"""
    from_team = event.get('from_team', 'Unknown')
    to_team = event.get('to_team', 'Unknown')
    pass_type = event.get('pass_type', 'Unknown')
    result = event.get('result', 'Unknown')
    confidence = event.get('confidence', 70)
    time_seconds = event.get('frame', 0) / fps
    
    from_class = from_team.lower() if from_team in ['Blue', 'Red'] else 'unknown'
    to_class = to_team.lower() if to_team in ['Blue', 'Red'] else 'unknown'
    result_class = result.lower() if result in ['Success', 'Fail'] else 'unknown'
    pass_color = PASS_COLORS.get(pass_type, '#757575')
    
    conf_color = '#22c55e' if confidence >= 80 else ('#f59e0b' if confidence >= 60 else '#ef4444')
    
    time_str = f"{int(time_seconds//60)}:{int(time_seconds%60):02d}"
    pass_type_js = pass_type.replace("'", "\\'")
    
    return f'''                            <tr data-team="{from_team}" data-result="{result}" data-type="{pass_type}" onclick="openVideoPopup('pass', {time_seconds:.2f}, '{pass_type_js}', '{from_team}', '{event.get('from_player', 0)}', '{to_team}', '{event.get('to_player', 0)}', '{result}', {confidence})">
                                <td>{idx}</td>
                                <td><button class="timestamp-btn">▶ {time_str}</button></td>
                                <td>
                                    <div class="player-cell">
                                        <div class="player-avatar {from_class}">#{event.get('from_player', '?')}</div>
                                        <span>{from_team}</span>
                                    </div>
                                </td>
                                <td>
                                    <div class="player-cell">
                                        <div class="player-avatar {to_class}">#{event.get('to_player', '?')}</div>
                                        <span>{to_team}</span>
                                    </div>
                                </td>
                                <td><span class="type-tag" style="background: {pass_color};">{pass_type}</span></td>
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


def generate_shot_row(idx, event, fps):
    """Generate HTML table row for a shot event"""
    team = event.get('team', 'Unknown')
    shot_type = event.get('shot_type', 'Unknown')
    confidence = event.get('confidence', 70)
    time_seconds = event.get('frame', 0) / fps
    
    team_class = team.lower() if team in ['Blue', 'Red'] else 'unknown'
    shot_color = SHOT_COLORS.get(shot_type, '#757575')
    
    conf_color = '#22c55e' if confidence >= 80 else ('#f59e0b' if confidence >= 60 else '#ef4444')
    
    time_str = f"{int(time_seconds//60)}:{int(time_seconds%60):02d}"
    shot_type_js = shot_type.replace("'", "\\'")
    
    result = 'Goal!' if shot_type == 'Goal' else ('Saved' if shot_type == 'Shot on target' else 'Miss')
    
    return f'''                            <tr data-team="{team}" data-type="{shot_type}" onclick="openVideoPopup('shot', {time_seconds:.2f}, '{shot_type_js}', '{team}', '{event.get('shooter_id', 0)}', '', '', '{result}', {confidence})">
                                <td>{idx}</td>
                                <td><button class="timestamp-btn">▶ {time_str}</button></td>
                                <td>
                                    <div class="player-cell">
                                        <div class="player-avatar {team_class}">#{event.get('shooter_id', '?')}</div>
                                    </div>
                                </td>
                                <td><span style="color: {'#3b82f6' if team == 'Blue' else '#ef4444'}">{team}</span></td>
                                <td><span class="type-tag" style="background: {shot_color};">{shot_type}</span></td>
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


def generate_markers_js(pass_events, shot_events, fps):
    """Generate JavaScript array of timeline markers"""
    markers = []
    
    # Pass markers
    for idx, event in enumerate(pass_events, 1):
        time_seconds = event.get('frame', 0) / fps
        pass_type = event.get('pass_type', 'Unknown')
        color = PASS_COLORS.get(pass_type, '#757575')
        team = event.get('from_team', 'Unknown')
        markers.append({
            'time': time_seconds,
            'type': pass_type,
            'color': color,
            'team': team,
            'idx': idx,
            'eventType': 'pass'
        })
    
    # Shot markers
    for idx, event in enumerate(shot_events, 1):
        time_seconds = event.get('frame', 0) / fps
        shot_type = event.get('shot_type', 'Unknown')
        color = SHOT_COLORS.get(shot_type, '#757575')
        team = event.get('team', 'Unknown')
        markers.append({
            'time': time_seconds,
            'type': shot_type,
            'color': color,
            'team': team,
            'idx': idx + len(pass_events),
            'eventType': 'shot'
        })
    
    # Generate JavaScript
    js_lines = ["const allMarkers = ["]
    for m in markers:
        js_lines.append(f"    {{time: {m['time']:.2f}, type: \"{m['type']}\", color: \"{m['color']}\", team: \"{m['team']}\", idx: {m['idx']}, eventType: \"{m['eventType']}\"}},")
    js_lines.append("];")
    
    return "\n        ".join(js_lines)


# === STANDALONE TEST ===
if __name__ == "__main__":
    # Test with sample data
    sample_passes = [
        {"frame": 120, "from_player": 8, "to_player": 12, "from_team": "Blue", "to_team": "Red", "pass_type": "Short pass", "result": "Fail", "confidence": 82},
        {"frame": 300, "from_player": 12, "to_player": 2, "from_team": "Red", "to_team": "Red", "pass_type": "Long pass", "result": "Success", "confidence": 85},
    ]
    
    sample_shots = [
        {"frame": 450, "shooter_id": 9, "team": "Blue", "shot_type": "Shot on target", "confidence": 78},
        {"frame": 600, "shooter_id": 11, "team": "Red", "shot_type": "Goal", "confidence": 92},
    ]
    
    output = generate_full_scout_report_html(
        sample_passes, 
        sample_shots, 
        "test_video.mp4",
        "annotated_test_video.mp4",
        {}, {}, 
        30
    )
    print(f"✅ Test report generated: {output}")
