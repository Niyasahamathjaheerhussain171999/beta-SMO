# MLOps Guide - ScoutMe Pipeline

## Overview

This guide explains the MLOps (Machine Learning Operations) setup for the ScoutMe football analysis pipeline, including configuration, logging, auto-shutdown, and deployment on Lightning AI.

---

## 📁 Configuration File: `config.yaml`

The `config.yaml` file is your **Control Center** - change settings here without touching code.

### Location
```
smo-model-niyas/config.yaml
```

### Key Sections

#### 1. **Pipeline Settings**
```yaml
pipeline:
  run_name: "Match_Day_1_Test"
  input_folder: "data/input"
  output_folder: "data/output"
  save_annotated_video: true
  
  # Auto-shutdown (cost saving)
  auto_shutdown_enabled: true
  shutdown_delay_seconds: 10
```

#### 2. **Model Configuration**
```yaml
models:
  yolo_player: "data/football-player-detection.pt"
  yolo_ball: "data/football-ball-detection.pt"
  yolo_pitch: "data/football-pitch-detection.pt"
  qwen_model: "Qwen/Qwen2.5-VL-7B-Instruct"
  molmo_model: "allenai/Molmo-7B-D-0924"
```

#### 3. **Detection Thresholds**
```yaml
pass_detection:
  confidence_threshold: 75
  ball_proximity_threshold: 65
  # ... more settings

shot_detection:
  shot_velocity_threshold: 15
  goal_proximity_radius_meters: 35
  max_shot_angle_degrees: 45
  # ... more settings
```

#### 4. **Performance Mode**
```yaml
performance:
  smart_frame_processing: true
  skip_breaks: true
  process_all_active_play: true
  skip_annotated_video: true  # Faster processing
```

---

## 🔧 Auto-Shutdown Feature

### Purpose
Automatically shuts down Lightning AI Studio after job completion to **save GPU costs**.

### How It Works

1. **After Analysis Completes**: The `finally` block in `main4.py` executes
2. **Wait Period**: Waits `shutdown_delay_seconds` (default: 10 seconds)
3. **Shutdown Command**: Executes `studio stop` command
4. **Cost Savings**: Prevents GPU from running idle

### Configuration

```yaml
pipeline:
  auto_shutdown_enabled: true    # Set false to disable
  shutdown_delay_seconds: 10      # Wait time before shutdown
```

### Disable Auto-Shutdown

Set in `config.yaml`:
```yaml
pipeline:
  auto_shutdown_enabled: false
```

Or if config.yaml doesn't exist, it defaults to `True` (safe default).

---

## 📝 Logging

### Log File Location
```
logs/analysis_YYYYMMDD_HHMMSS.log
```

### Log Format
```
2026-01-08 13:23:09,975 - INFO - 🎬 STEP 4: STARTING VIDEO ANALYSIS
2026-01-08 13:23:10,047 - INFO - 🔍 Initializing automatic half detection...
```

### Log Levels
- **INFO**: General information
- **WARNING**: Non-critical issues
- **ERROR**: Errors that don't stop execution
- **CRITICAL**: Fatal errors

### Log Rotation
- Each run creates a new log file with timestamp
- Old logs are preserved (not auto-deleted)
- Manual cleanup recommended for long-term storage

---

## 🚀 Lightning AI Deployment

### Setup Script

Use `config/setup_lightning.sh` to set up the environment:

```bash
bash config/setup_lightning.sh
```

**What it does:**
1. Creates necessary directories
2. Updates pip
3. Installs dependencies from `requirements.txt`
4. Installs specialized packages (Qwen, flash-attn, yt-dlp)
5. Verifies installations

### Run Script

Use `config/run_in_lightning.sh` to run analysis:

```bash
bash config/run_in_lightning.sh
```

Or manually:
```bash
python src/main4.py --source "input_video/your_video.mp4" --debug
```

---

## ⚙️ Configuration Loading

### How It Works

1. **Check for config.yaml**: Looks in project root
2. **Load YAML**: Uses `yaml.safe_load()` to parse
3. **Fallback**: If file doesn't exist, uses defaults
4. **Error Handling**: Gracefully handles missing/invalid config

### Code Location

In `src/main4.py`:
```python
# Load configuration
cfg = {}
config_path = 'config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
else:
    # Use defaults
    cfg = {}
```

### Default Values

If `config.yaml` doesn't exist:
- `auto_shutdown_enabled`: `True` (safe default)
- `shutdown_delay_seconds`: `10`
- All other settings use hardcoded defaults in code

---

## 📊 MLOps Workflow

### Step 1: Configuration
- Load `config.yaml` (if exists)
- Set up logging
- Initialize models

### Step 2: Analysis
- Process video
- Detect passes, shots, events
- Generate statistics

### Step 3: Auto-Shutdown (MLOps)
- Wait for delay period
- Execute shutdown command
- Save logs

### Step 4: Output
- Save JSON reports
- Save CSV files
- Generate HTML reports
- Save annotated video (if enabled)

---

## 🔍 Troubleshooting

### Config Not Loading

**Problem**: Changes in `config.yaml` not taking effect

**Solution**:
1. Check file location: Must be in project root
2. Check YAML syntax: Use online YAML validator
3. Check file permissions: Ensure readable
4. Check logs: Look for config loading messages

### Auto-Shutdown Not Working

**Problem**: Studio doesn't shut down after job

**Solution**:
1. Check `auto_shutdown_enabled` in config
2. Check Lightning AI permissions
3. Check logs for shutdown command output
4. Manually stop: `studio stop`

### Logs Not Saving

**Problem**: No log files created

**Solution**:
1. Check `logs/` directory exists
2. Check write permissions
3. Check disk space
4. Check logs for file creation errors

---

## 📋 Best Practices

### 1. **Version Control**
- ✅ Commit `config.yaml` to git (with defaults)
- ❌ Don't commit sensitive API keys
- ✅ Use environment variables for secrets

### 2. **Configuration Management**
- ✅ Use `config.yaml` for all settings
- ✅ Document all config options
- ✅ Use meaningful default values

### 3. **Logging**
- ✅ Enable logging for production
- ✅ Review logs after each run
- ✅ Archive old logs periodically

### 4. **Cost Management**
- ✅ Enable auto-shutdown in production
- ✅ Monitor GPU usage
- ✅ Set appropriate shutdown delays

### 5. **Error Handling**
- ✅ Graceful fallbacks for missing config
- ✅ Log all errors
- ✅ Continue execution when possible

---

## 🎯 Quick Reference

### Enable Auto-Shutdown
```yaml
pipeline:
  auto_shutdown_enabled: true
```

### Disable Auto-Shutdown
```yaml
pipeline:
  auto_shutdown_enabled: false
```

### Change Shutdown Delay
```yaml
pipeline:
  shutdown_delay_seconds: 30  # Wait 30 seconds
```

### Skip Annotated Video (Faster)
```yaml
performance:
  skip_annotated_video: true
```

### Enable Debug Logging
```bash
python src/main4.py --source video.mp4 --debug
```

---

## 📚 Related Files

- **`config.yaml`**: Main configuration file
- **`src/main4.py`**: Main pipeline code (MLOps integration)
- **`config/setup_lightning.sh`**: Lightning AI setup script
- **`config/run_in_lightning.sh`**: Lightning AI run script
- **`requirements.txt`**: Python dependencies

---

## 🔗 Additional Resources

- [Lightning AI Documentation](https://lightning.ai/docs)
- [YAML Syntax Guide](https://yaml.org/spec/)
- [Python Logging Guide](https://docs.python.org/3/library/logging.html)

---

**Your MLOps setup is now complete and ready for production!** 🚀
