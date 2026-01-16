"""
ScoutMe - Utility Modules

Video Analysis Utilities:
- video_health_check: Diagnose video file issues
- video_processor: Repair and enhance videos
- model_validator: Validate AI model configurations
- pitch_calibrator: Convert pixel coordinates to real-world meters
"""

from .video_health_check import VideoHealthChecker
from .video_processor import VideoProcessor
from .model_validator import ModelValidator, validate_models_from_config
from .pitch_calibrator import (
    PitchCalibrator,
    get_calibrator,
    calibrate_pitch,
    pixel_to_meters,
    distance_to_goal
)

__all__ = [
    'VideoHealthChecker',
    'VideoProcessor',
    'ModelValidator',
    'validate_models_from_config',
    'PitchCalibrator',
    'get_calibrator',
    'calibrate_pitch',
    'pixel_to_meters',
    'distance_to_goal',
]

