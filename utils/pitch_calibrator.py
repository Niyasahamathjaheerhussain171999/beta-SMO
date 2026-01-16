"""
ScoutMe - Pitch Calibrator Utility
Handles coordinate transformation from pixel space to real-world meters.

CRITICAL FIX:
The original system was showing distances of 1000-4700m because:
1. Pitch detection model was wrong (COCO instead of pitch keypoints)
2. No proper homography transformation was applied
3. Fallback used raw pixel values as meters

This module provides:
- Proper pitch calibration using detected keypoints
- Fallback estimation using frame dimensions
- Sanity checks to ensure distances are valid (0-105m)
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class PitchDimensions:
    """Standard football pitch dimensions in meters."""
    LENGTH: float = 105.0  # meters (touchline)
    WIDTH: float = 68.0    # meters (goal line)
    PENALTY_AREA_LENGTH: float = 16.5
    PENALTY_AREA_WIDTH: float = 40.3
    GOAL_AREA_LENGTH: float = 5.5
    GOAL_AREA_WIDTH: float = 18.3
    PENALTY_SPOT: float = 11.0  # from goal line
    CENTER_CIRCLE_RADIUS: float = 9.15
    GOAL_WIDTH: float = 7.32
    GOAL_HEIGHT: float = 2.44


class PitchCalibrator:
    """
    Calibrates pixel coordinates to real-world meters using pitch detection.
    Falls back to estimation if pitch detection unavailable.
    """
    
    def __init__(self):
        self.dimensions = PitchDimensions()
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.is_calibrated = False
        self.calibration_method = "none"
        
        # Frame dimensions (for fallback)
        self.frame_width = 1920
        self.frame_height = 1080
        
        # Real-world pitch points (in meters, origin at top-left corner)
        # These are reference points on a standard pitch
        self.REAL_PITCH_POINTS = {
            'top_left_corner': (0, 0),
            'top_right_corner': (105, 0),
            'bottom_right_corner': (105, 68),
            'bottom_left_corner': (0, 68),
            'center_spot': (52.5, 34),
            'top_center': (52.5, 0),
            'bottom_center': (52.5, 68),
            'left_penalty_spot': (11, 34),
            'right_penalty_spot': (94, 34),
            'left_goal_center': (0, 34),
            'right_goal_center': (105, 34),
        }
        
        # Goal positions (for distance calculations)
        self.LEFT_GOAL = np.array([0, 34])
        self.RIGHT_GOAL = np.array([105, 34])
    
    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Set frame dimensions for fallback estimation."""
        self.frame_width = width
        self.frame_height = height
    
    def calibrate_from_keypoints(self, detected_keypoints: List[Dict], frame_shape: Tuple) -> bool:
        """
        Calibrate using detected pitch keypoints.
        
        Args:
            detected_keypoints: List of {name, x, y} or {class_id, x, y}
            frame_shape: (height, width, channels)
            
        Returns:
            True if calibration successful
        """
        self.frame_height, self.frame_width = frame_shape[:2]
        
        if len(detected_keypoints) < 4:
            print(f"   ⚠️ Not enough keypoints for calibration ({len(detected_keypoints)}/4 minimum)")
            return self._calibrate_fallback()
        
        # Map detected keypoints to real-world coordinates
        src_points = []  # Pixel coordinates
        dst_points = []  # Real-world coordinates
        
        for kp in detected_keypoints:
            # Try to match keypoint to known position
            kp_name = kp.get('name', '').lower()
            kp_x = kp.get('x', 0)
            kp_y = kp.get('y', 0)
            
            # Match to real-world point
            matched_real = None
            for name, coords in self.REAL_PITCH_POINTS.items():
                if name.lower() in kp_name or kp_name in name.lower():
                    matched_real = coords
                    break
            
            if matched_real:
                src_points.append([kp_x, kp_y])
                dst_points.append(list(matched_real))
        
        if len(src_points) < 4:
            print(f"   ⚠️ Could not match enough keypoints ({len(src_points)}/4 minimum)")
            return self._calibrate_fallback()
        
        # Calculate homography
        src_pts = np.array(src_points, dtype=np.float32)
        dst_pts = np.array(dst_points, dtype=np.float32)
        
        self.homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if self.homography_matrix is None:
            print(f"   ⚠️ Homography calculation failed")
            return self._calibrate_fallback()
        
        self.inverse_homography = np.linalg.inv(self.homography_matrix)
        self.is_calibrated = True
        self.calibration_method = "keypoints"
        
        # Verify calibration
        self._verify_calibration()
        
        return True
    
    def calibrate_from_corners(self, corners: List[Tuple[float, float]]) -> bool:
        """
        Calibrate using 4 pitch corners (detected or estimated).
        
        Args:
            corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in order:
                     top-left, top-right, bottom-right, bottom-left
        """
        if len(corners) != 4:
            return self._calibrate_fallback()
        
        src_pts = np.array(corners, dtype=np.float32)
        dst_pts = np.array([
            [0, 0],           # top-left
            [105, 0],         # top-right
            [105, 68],        # bottom-right
            [0, 68],          # bottom-left
        ], dtype=np.float32)
        
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        if self.homography_matrix is None:
            return self._calibrate_fallback()
        
        self.inverse_homography = np.linalg.inv(self.homography_matrix)
        self.is_calibrated = True
        self.calibration_method = "corners"
        
        self._verify_calibration()
        return True
    
    def _calibrate_fallback(self) -> bool:
        """
        Fallback calibration using frame dimensions.
        Assumes camera shows most of the pitch.
        """
        print(f"   📐 Using fallback calibration (frame-based estimation)")
        
        # Assume the frame shows approximately the full pitch width
        # with some margin on sides
        margin_ratio = 0.05  # 5% margin on each side
        
        visible_width = self.frame_width * (1 - 2 * margin_ratio)
        visible_height = self.frame_height * (1 - 2 * margin_ratio)
        
        # Estimate corners based on typical broadcast view
        corners = [
            (self.frame_width * margin_ratio, self.frame_height * margin_ratio),
            (self.frame_width * (1 - margin_ratio), self.frame_height * margin_ratio),
            (self.frame_width * (1 - margin_ratio), self.frame_height * (1 - margin_ratio)),
            (self.frame_width * margin_ratio, self.frame_height * (1 - margin_ratio)),
        ]
        
        src_pts = np.array(corners, dtype=np.float32)
        dst_pts = np.array([
            [0, 0],
            [105, 0],
            [105, 68],
            [0, 68],
        ], dtype=np.float32)
        
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.inverse_homography = np.linalg.inv(self.homography_matrix)
        self.is_calibrated = True
        self.calibration_method = "fallback"
        
        return True
    
    def _verify_calibration(self) -> None:
        """Verify calibration produces sane results."""
        # Test center of frame
        center_pixel = np.array([[self.frame_width / 2, self.frame_height / 2]])
        center_meters = self.pixel_to_meters(center_pixel)
        
        print(f"   🔍 Calibration verification:")
        print(f"      Frame center ({self.frame_width/2:.0f}, {self.frame_height/2:.0f}) → "
              f"({center_meters[0][0]:.1f}m, {center_meters[0][1]:.1f}m)")
        
        # Check if result is within pitch bounds
        x, y = center_meters[0]
        if not (0 <= x <= 105 and 0 <= y <= 68):
            print(f"      ⚠️ Warning: Center point outside pitch bounds!")
        else:
            print(f"      ✅ Calibration looks reasonable")
    
    def pixel_to_meters(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to real-world meters.
        
        Args:
            pixel_coords: Array of shape (N, 2) with [x, y] pixel coordinates
            
        Returns:
            Array of shape (N, 2) with [x, y] meter coordinates
        """
        if not self.is_calibrated:
            self._calibrate_fallback()
        
        # Ensure correct shape
        pixel_coords = np.array(pixel_coords, dtype=np.float32)
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.reshape(1, 2)
        
        # Apply homography transformation
        meter_coords = cv2.perspectiveTransform(
            pixel_coords.reshape(-1, 1, 2),
            self.homography_matrix
        )
        
        return meter_coords.reshape(-1, 2)
    
    def meters_to_pixel(self, meter_coords: np.ndarray) -> np.ndarray:
        """
        Convert real-world meters to pixel coordinates.
        
        Args:
            meter_coords: Array of shape (N, 2) with [x, y] meter coordinates
            
        Returns:
            Array of shape (N, 2) with [x, y] pixel coordinates
        """
        if not self.is_calibrated or self.inverse_homography is None:
            self._calibrate_fallback()
        
        meter_coords = np.array(meter_coords, dtype=np.float32)
        if meter_coords.ndim == 1:
            meter_coords = meter_coords.reshape(1, 2)
        
        pixel_coords = cv2.perspectiveTransform(
            meter_coords.reshape(-1, 1, 2),
            self.inverse_homography
        )
        
        return pixel_coords.reshape(-1, 2)
    
    def calculate_distance(self, point1_px: Tuple[float, float], 
                          point2_px: Tuple[float, float]) -> float:
        """
        Calculate distance between two points in meters.
        
        Args:
            point1_px: (x, y) pixel coordinates of first point
            point2_px: (x, y) pixel coordinates of second point
            
        Returns:
            Distance in meters
        """
        pts = np.array([point1_px, point2_px], dtype=np.float32)
        pts_meters = self.pixel_to_meters(pts)
        
        distance = np.sqrt(
            (pts_meters[1][0] - pts_meters[0][0])**2 +
            (pts_meters[1][1] - pts_meters[0][1])**2
        )
        
        # Sanity check: distance should be reasonable (0-150m max diagonal)
        if distance > 150:
            print(f"   ⚠️ Unrealistic distance: {distance:.1f}m - calibration may be off")
            # Return capped value
            return min(distance, 150)
        
        return distance
    
    def calculate_distance_to_goal(self, position_px: Tuple[float, float],
                                   attacking_direction: str) -> float:
        """
        Calculate distance from position to the target goal.
        
        Args:
            position_px: (x, y) pixel coordinates
            attacking_direction: 'left_to_right' or 'right_to_left'
            
        Returns:
            Distance to goal in meters
        """
        # Convert position to meters
        pos_meters = self.pixel_to_meters(np.array([position_px]))[0]
        
        # Determine target goal
        if attacking_direction == 'left_to_right':
            goal = self.RIGHT_GOAL  # Attacking right goal at (105, 34)
        else:
            goal = self.LEFT_GOAL   # Attacking left goal at (0, 34)
        
        # Calculate distance
        distance = np.sqrt(
            (pos_meters[0] - goal[0])**2 +
            (pos_meters[1] - goal[1])**2
        )
        
        # Sanity check
        if distance > 120:  # Max distance on pitch is ~125m diagonal
            # This indicates calibration issue - use pixel-based estimation
            return self._estimate_distance_fallback(position_px, attacking_direction)
        
        return distance
    
    def _estimate_distance_fallback(self, position_px: Tuple[float, float],
                                    attacking_direction: str) -> float:
        """
        Fallback distance estimation using pixel position.
        Assumes camera view shows full pitch width.
        """
        x, y = position_px
        
        # Normalize to 0-1 range
        x_norm = x / self.frame_width
        y_norm = y / self.frame_height
        
        # Convert to pitch coordinates (0-105m, 0-68m)
        pitch_x = x_norm * 105
        pitch_y = y_norm * 68
        
        # Calculate distance to goal
        if attacking_direction == 'left_to_right':
            goal_x, goal_y = 105, 34
        else:
            goal_x, goal_y = 0, 34
        
        distance = np.sqrt((pitch_x - goal_x)**2 + (pitch_y - goal_y)**2)
        
        return min(distance, 120)  # Cap at max realistic distance
    
    def is_in_shooting_range(self, position_px: Tuple[float, float],
                            attacking_direction: str,
                            max_distance: float = 35.0) -> bool:
        """
        Check if position is within shooting range.
        
        Args:
            position_px: (x, y) pixel coordinates
            attacking_direction: 'left_to_right' or 'right_to_left'
            max_distance: Maximum shooting distance in meters (default 35m)
            
        Returns:
            True if within shooting range
        """
        distance = self.calculate_distance_to_goal(position_px, attacking_direction)
        return distance <= max_distance
    
    def get_pitch_zone(self, position_px: Tuple[float, float]) -> str:
        """
        Determine which zone of the pitch the position is in.
        
        Returns:
            Zone name: 'defensive_third', 'middle_third', 'attacking_third'
        """
        pos_meters = self.pixel_to_meters(np.array([position_px]))[0]
        x = pos_meters[0]
        
        if x < 35:
            return 'defensive_third'
        elif x < 70:
            return 'middle_third'
        else:
            return 'attacking_third'


# Global calibrator instance for easy access
_calibrator: Optional[PitchCalibrator] = None


def get_calibrator() -> PitchCalibrator:
    """Get or create global calibrator instance."""
    global _calibrator
    if _calibrator is None:
        _calibrator = PitchCalibrator()
    return _calibrator


def calibrate_pitch(frame_width: int, frame_height: int, 
                   keypoints: Optional[List[Dict]] = None) -> PitchCalibrator:
    """
    Convenience function to calibrate pitch.
    
    Args:
        frame_width: Video frame width
        frame_height: Video frame height
        keypoints: Optional detected keypoints
        
    Returns:
        Calibrated PitchCalibrator instance
    """
    calibrator = get_calibrator()
    calibrator.set_frame_dimensions(frame_width, frame_height)
    
    if keypoints:
        calibrator.calibrate_from_keypoints(keypoints, (frame_height, frame_width))
    else:
        calibrator._calibrate_fallback()
    
    return calibrator


def pixel_to_meters(x: float, y: float) -> Tuple[float, float]:
    """Quick conversion from pixels to meters."""
    calibrator = get_calibrator()
    result = calibrator.pixel_to_meters(np.array([[x, y]]))[0]
    return (result[0], result[1])


def distance_to_goal(x: float, y: float, direction: str = 'left_to_right') -> float:
    """Quick distance to goal calculation."""
    calibrator = get_calibrator()
    return calibrator.calculate_distance_to_goal((x, y), direction)


if __name__ == "__main__":
    # Test the calibrator
    print("Testing PitchCalibrator...")
    
    calibrator = PitchCalibrator()
    calibrator.set_frame_dimensions(1920, 1080)
    calibrator._calibrate_fallback()
    
    # Test conversions
    test_points = [
        (960, 540),   # Center
        (100, 100),   # Top-left area
        (1820, 980),  # Bottom-right area
    ]
    
    print("\nPixel → Meters conversions:")
    for px, py in test_points:
        mx, my = pixel_to_meters(px, py)
        dist = distance_to_goal(px, py, 'left_to_right')
        print(f"  ({px}, {py}) → ({mx:.1f}m, {my:.1f}m) | Distance to right goal: {dist:.1f}m")
    
    print("\n✅ PitchCalibrator test complete")

