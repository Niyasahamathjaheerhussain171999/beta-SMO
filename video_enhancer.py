"""
ScoutMe - Video Enhancement Module
Pre-processes Veo footage for better detection accuracy

STRATEGY:
1. Denoising - Remove compression artifacts
2. Contrast Enhancement - Improve jersey color separation
3. Stabilization - Reduce camera shake (optional)
4. Super-Resolution - Upscale for better small object detection (optional)

NOTE: This module provides OPTIONAL pre-processing.
For most Veo footage, the existing pipeline works well.
Use enhancement only for problematic footage.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os


class VideoEnhancer:
    """
    Enhances video quality for better AI detection accuracy.
    Optimized for Veo camera footage (1080p, 50fps).
    """
    
    def __init__(self, enhancement_level: str = "medium"):
        """
        Initialize video enhancer.
        
        Args:
            enhancement_level: "light", "medium", or "heavy"
        """
        self.enhancement_level = enhancement_level
        
        # Enhancement parameters based on level
        self.params = {
            "light": {
                "denoise_h": 3,
                "denoise_template": 7,
                "denoise_search": 21,
                "clahe_clip": 1.5,
                "clahe_grid": (8, 8),
                "sharpen": False
            },
            "medium": {
                "denoise_h": 5,
                "denoise_template": 7,
                "denoise_search": 21,
                "clahe_clip": 2.0,
                "clahe_grid": (8, 8),
                "sharpen": True,
                "sharpen_kernel": np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 5
            },
            "heavy": {
                "denoise_h": 10,
                "denoise_template": 7,
                "denoise_search": 21,
                "clahe_clip": 3.0,
                "clahe_grid": (4, 4),
                "sharpen": True,
                "sharpen_kernel": np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]]) / 6
            }
        }
        
        self.current_params = self.params.get(enhancement_level, self.params["medium"])
        
        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=self.current_params["clahe_clip"],
            tileGridSize=self.current_params["clahe_grid"]
        )
        
        # Stats
        self.frames_processed = 0
    
    def denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply denoising to reduce compression artifacts.
        Uses Non-local Means Denoising for color images.
        """
        return cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=self.current_params["denoise_h"],
            hForColorComponents=self.current_params["denoise_h"],
            templateWindowSize=self.current_params["denoise_template"],
            searchWindowSize=self.current_params["denoise_search"]
        )
    
    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        This improves jersey color separation and ball visibility.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def sharpen_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply sharpening to improve edge detection.
        """
        if not self.current_params.get("sharpen", False):
            return frame
        
        kernel = self.current_params["sharpen_kernel"]
        return cv2.filter2D(frame, -1, kernel)
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all enhancements to a single frame.
        
        Order: Denoise → Contrast → Sharpen
        """
        self.frames_processed += 1
        
        # Step 1: Denoise
        enhanced = self.denoise_frame(frame)
        
        # Step 2: Contrast enhancement
        enhanced = self.enhance_contrast(enhanced)
        
        # Step 3: Sharpen (if enabled)
        enhanced = self.sharpen_frame(enhanced)
        
        return enhanced
    
    def enhance_video(self, input_path: str, output_path: str, 
                      progress_callback=None) -> bool:
        """
        Enhance entire video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to save enhanced video
            progress_callback: Optional callback(current_frame, total_frames)
        
        Returns: True if successful
        """
        try:
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                print(f"❌ Cannot open video: {input_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Enhancing video: {input_path}")
            print(f"   Resolution: {width}x{height} @ {fps:.1f}fps")
            print(f"   Total frames: {total_frames}")
            print(f"   Enhancement level: {self.enhancement_level}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhance frame
                enhanced = self.enhance_frame(frame)
                writer.write(enhanced)
                
                frame_idx += 1
                
                if progress_callback:
                    progress_callback(frame_idx, total_frames)
                
                if frame_idx % 500 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"   Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
            
            cap.release()
            writer.release()
            
            print(f"✅ Enhanced video saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Video enhancement error: {e}")
            return False


class VeoOptimizer:
    """
    Optimizes settings specifically for Veo camera footage.
    Detects Veo footage and adjusts enhancement parameters.
    """
    
    @staticmethod
    def detect_veo_footage(video_path: str) -> Tuple[bool, dict]:
        """
        Detect if video is from Veo camera based on:
        1. FPS (Veo typically 50fps)
        2. Resolution (Veo typically 1080p or 4K)
        3. Bitrate patterns
        
        Returns: (is_veo, metadata)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            metadata = {
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "duration_seconds": total_frames / fps if fps > 0 else 0
            }
            
            # Veo detection heuristics
            is_veo = False
            veo_score = 0
            
            # Check FPS (Veo is typically 50fps)
            if 48 <= fps <= 52:
                veo_score += 3
            elif 45 <= fps <= 55:
                veo_score += 1
            
            # Check resolution (1080p or higher)
            if height >= 1080:
                veo_score += 1
            
            # Check duration (match length)
            if 60 * 60 <= metadata["duration_seconds"] <= 3 * 60 * 60:  # 1-3 hours
                veo_score += 1
            
            is_veo = veo_score >= 3
            metadata["is_veo"] = is_veo
            metadata["veo_score"] = veo_score
            
            return is_veo, metadata
            
        except Exception as e:
            print(f"Veo detection error: {e}")
            return False, {}
    
    @staticmethod
    def get_optimal_settings(metadata: dict) -> dict:
        """
        Get optimal processing settings for the detected video.
        """
        settings = {
            "enhancement_level": "light",  # Default
            "skip_enhancement": False,
            "process_every_n_frames": 1,
            "recommended_imgsz": 1280
        }
        
        fps = metadata.get("fps", 25)
        height = metadata.get("height", 720)
        is_veo = metadata.get("is_veo", False)
        
        if is_veo:
            # Veo footage is usually high quality
            settings["enhancement_level"] = "light"
            settings["skip_enhancement"] = True  # Often not needed
            
            # At 50fps, can process fewer frames for speed
            if fps >= 50:
                settings["process_every_n_frames"] = 2  # Skip every other frame
        
        else:
            # Non-Veo footage may need more enhancement
            if height < 720:
                settings["enhancement_level"] = "heavy"
            elif height < 1080:
                settings["enhancement_level"] = "medium"
            else:
                settings["enhancement_level"] = "light"
        
        # Adjust image size based on resolution
        if height >= 2160:  # 4K
            settings["recommended_imgsz"] = 1920
        elif height >= 1080:
            settings["recommended_imgsz"] = 1280
        else:
            settings["recommended_imgsz"] = 960
        
        return settings


def enhance_for_analysis(input_path: str, output_path: str = None,
                        enhancement_level: str = "auto") -> str:
    """
    Convenience function to enhance video for analysis.
    
    Args:
        input_path: Path to input video
        output_path: Optional output path (auto-generated if None)
        enhancement_level: "auto", "light", "medium", "heavy", or "skip"
    
    Returns: Path to enhanced video (or original if skipped)
    """
    # Auto-detect Veo and get optimal settings
    is_veo, metadata = VeoOptimizer.detect_veo_footage(input_path)
    
    print(f"📊 Video Analysis:")
    print(f"   Veo Camera: {'Yes' if is_veo else 'No'}")
    print(f"   FPS: {metadata.get('fps', 'Unknown')}")
    print(f"   Resolution: {metadata.get('width', '?')}x{metadata.get('height', '?')}")
    
    if enhancement_level == "auto":
        settings = VeoOptimizer.get_optimal_settings(metadata)
        
        if settings["skip_enhancement"]:
            print(f"   ✅ High-quality footage detected - skipping enhancement")
            return input_path
        
        enhancement_level = settings["enhancement_level"]
    
    if enhancement_level == "skip":
        return input_path
    
    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_enhanced{ext}"
    
    # Enhance video
    enhancer = VideoEnhancer(enhancement_level)
    success = enhancer.enhance_video(input_path, output_path)
    
    if success:
        return output_path
    else:
        return input_path


# === BALL-SPECIFIC ENHANCEMENT ===

def enhance_ball_region(frame: np.ndarray, ball_xy: Tuple[int, int], 
                        crop_size: int = 200) -> np.ndarray:
    """
    Apply focused enhancement around ball region for better tracking.
    """
    h, w = frame.shape[:2]
    x, y = int(ball_xy[0]), int(ball_xy[1])
    
    # Extract region
    x1 = max(0, x - crop_size // 2)
    y1 = max(0, y - crop_size // 2)
    x2 = min(w, x + crop_size // 2)
    y2 = min(h, y + crop_size // 2)
    
    region = frame[y1:y2, x1:x2].copy()
    
    if region.size == 0:
        return frame
    
    # Apply stronger enhancement to ball region
    # Increase contrast to make ball stand out
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced_region = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Sharpen ball region
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 4
    enhanced_region = cv2.filter2D(enhanced_region, -1, kernel)
    
    # Blend back into frame
    result = frame.copy()
    result[y1:y2, x1:x2] = enhanced_region
    
    return result
