"""
ScoutMe - Video Processor & Repair Utility
Repairs corrupted videos and enhances quality for AI processing.

USAGE:
    python utils/video_processor.py path/to/video.mp4 --mode repair
    python utils/video_processor.py path/to/video.mp4 --mode enhance

MODES:
- repair: Fast fix for corrupted videos (copy streams, fix moov atom)
- enhance: AI-optimized preprocessing (denoise, sharpen, color correction)
"""

import subprocess
import shutil
import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import cv2


class VideoProcessor:
    """
    Video repair and enhancement pipeline.
    Fixes corruption and optimizes videos for AI detection.
    """
    
    def __init__(self, input_path: str, output_path: Optional[str] = None):
        self.input_path = Path(input_path)
        
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.input_path.parent / f"{self.input_path.stem}_processed.mp4"
        
        self.backup_path = self.input_path.parent / f"{self.input_path.stem}_original.mp4"
        self.temp_path = self.input_path.parent / f"{self.input_path.stem}_temp.mp4"
        
        # Check ffmpeg availability
        self.ffmpeg_available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def backup_original(self) -> bool:
        """Create backup of original video."""
        if self.backup_path.exists():
            print(f"   ⚠️ Backup already exists: {self.backup_path}")
            return True
        
        try:
            print(f"   📦 Creating backup: {self.backup_path}")
            shutil.copy2(self.input_path, self.backup_path)
            return True
        except Exception as e:
            print(f"   ❌ Backup failed: {e}")
            return False
    
    def repair_video(self) -> Optional[Path]:
        """
        Repair corrupted video using ffmpeg.
        - Fix moov atom position (faststart)
        - Re-mux without re-encoding (fast)
        - Handle truncated files gracefully
        """
        if not self.ffmpeg_available:
            print("   ❌ ffmpeg not available - cannot repair")
            return None
        
        print(f"   🔧 Repairing video...")
        
        # Method 1: Try simple copy with faststart
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', str(self.input_path),
            '-c', 'copy',  # Copy streams without re-encoding
            '-movflags', '+faststart',  # Move moov atom to start
            '-err_detect', 'ignore_err',  # Ignore errors
            str(self.temp_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0 and self.temp_path.exists():
                # Verify output
                if self._verify_video(self.temp_path):
                    shutil.move(self.temp_path, self.output_path)
                    print(f"   ✅ Repair successful (stream copy)")
                    return self.output_path
            
            # Method 2: Try re-encoding if copy failed
            print(f"   ⚠️ Stream copy failed, trying re-encode...")
            return self._repair_with_reencode()
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ Repair timed out")
            return None
        except Exception as e:
            print(f"   ❌ Repair failed: {e}")
            return None
    
    def _repair_with_reencode(self) -> Optional[Path]:
        """Re-encode video to fix corruption."""
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(self.input_path),
            '-c:v', 'libx264',
            '-preset', 'fast',  # Balance speed/quality
            '-crf', '20',  # Good quality
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-err_detect', 'ignore_err',
            str(self.temp_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for re-encoding
            )
            
            if result.returncode == 0 and self.temp_path.exists():
                if self._verify_video(self.temp_path):
                    shutil.move(self.temp_path, self.output_path)
                    print(f"   ✅ Repair successful (re-encoded)")
                    return self.output_path
            
            print(f"   ❌ Re-encode failed")
            return None
            
        except Exception as e:
            print(f"   ❌ Re-encode failed: {e}")
            return None
    
    def enhance_for_ai(self) -> Optional[Path]:
        """
        Apply AI-optimized preprocessing.
        - Denoise: Reduce camera sensor noise
        - Sharpen: Improve edge detection for players/ball
        - Color correction: Stabilize colors for team detection
        - Contrast enhancement: Make players stand out
        """
        if not self.ffmpeg_available:
            print("   ❌ ffmpeg not available - cannot enhance")
            return None
        
        print(f"   🎨 Enhancing video for AI detection...")
        print(f"   ⏳ This may take 30-60 minutes for a 2-hour video...")
        
        # Build filter chain
        filters = [
            # Denoise: Remove camera sensor noise (gentle)
            'hqdn3d=2:1.5:3:2.5',
            # Unsharp mask: Sharpen edges (players, ball, lines)
            'unsharp=3:3:0.8:3:3:0.0',
            # Color correction: Slight contrast and saturation boost
            'eq=contrast=1.05:brightness=0.01:saturation=1.03',
        ]
        
        filter_chain = ','.join(filters)
        
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(self.input_path),
            '-vf', filter_chain,
            '-c:v', 'libx264',
            '-preset', 'slow',  # Better quality
            '-crf', '18',  # High quality (visually lossless)
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-movflags', '+faststart',
            str(self.temp_path)
        ]
        
        try:
            print(f"   Running ffmpeg enhancement pipeline...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0 and self.temp_path.exists():
                if self._verify_video(self.temp_path):
                    shutil.move(self.temp_path, self.output_path)
                    print(f"   ✅ Enhancement successful")
                    return self.output_path
            
            print(f"   ❌ Enhancement failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:500]}")
            return None
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ Enhancement timed out (exceeded 2 hours)")
            return None
        except Exception as e:
            print(f"   ❌ Enhancement failed: {e}")
            return None
    
    def _verify_video(self, video_path: Path) -> bool:
        """Verify the processed video is valid."""
        if not video_path.exists():
            return False
        
        # Check file size (should be at least 1MB)
        if video_path.stat().st_size < 1024 * 1024:
            return False
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    
    def process(self, mode: str = 'repair', create_backup: bool = True) -> Optional[Path]:
        """
        Main processing pipeline.
        
        Args:
            mode: 'repair' (fast, just fix corruption) or 'enhance' (slow, improve quality)
            create_backup: Whether to backup original file
            
        Returns:
            Path to processed video, or None if failed
        """
        print(f"\n{'='*70}")
        print(f"🎬 VIDEO PROCESSOR")
        print(f"{'='*70}")
        print(f"   Input: {self.input_path}")
        print(f"   Output: {self.output_path}")
        print(f"   Mode: {mode}")
        print(f"{'='*70}\n")
        
        if not self.input_path.exists():
            print(f"   ❌ Input file not found: {self.input_path}")
            return None
        
        if not self.ffmpeg_available:
            print(f"   ❌ ffmpeg not available")
            print(f"   💡 Install ffmpeg:")
            print(f"      Linux: apt-get install ffmpeg")
            print(f"      Mac: brew install ffmpeg")
            print(f"      Windows: Download from ffmpeg.org")
            return None
        
        # Step 1: Backup
        if create_backup:
            if not self.backup_original():
                print(f"   ⚠️ Continuing without backup...")
        
        # Step 2: Process
        if mode == 'repair':
            result = self.repair_video()
        elif mode == 'enhance':
            result = self.enhance_for_ai()
        else:
            print(f"   ❌ Unknown mode: {mode}")
            return None
        
        # Step 3: Verify
        if result and result.exists():
            # Get file sizes for comparison
            original_size = self.input_path.stat().st_size / (1024 * 1024)
            processed_size = result.stat().st_size / (1024 * 1024)
            
            print(f"\n{'='*70}")
            print(f"✅ VIDEO PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"   Original: {original_size:.1f} MB")
            print(f"   Processed: {processed_size:.1f} MB")
            print(f"   Output: {result}")
            print(f"{'='*70}\n")
            
            return result
        else:
            print(f"\n{'='*70}")
            print(f"❌ VIDEO PROCESSING FAILED")
            print(f"{'='*70}")
            print(f"   💡 Try re-downloading the video from source")
            print(f"{'='*70}\n")
            return None
    
    def cleanup(self) -> None:
        """Remove temporary files."""
        if self.temp_path.exists():
            self.temp_path.unlink()


def convert_to_playable(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Quick function to convert video to web-playable format.
    Useful for HTML report playback.
    """
    input_p = Path(input_path)
    
    if output_path:
        output_p = Path(output_path)
    else:
        output_p = input_p.parent / f"{input_p.stem}_playable.mp4"
    
    cmd = [
        'ffmpeg',
        '-y',
        '-i', str(input_p),
        '-c:v', 'libx264',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        str(output_p)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0 and output_p.exists():
            return str(output_p)
    except Exception:
        pass
    
    return None


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Processor - Repair & Enhance")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--mode", choices=['repair', 'enhance'], default='repair',
                       help="Processing mode: 'repair' (fast) or 'enhance' (slow)")
    parser.add_argument("--output", "-o", help="Output path (default: input_processed.mp4)")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup")
    
    args = parser.parse_args()
    
    processor = VideoProcessor(args.video_path, args.output)
    result = processor.process(
        mode=args.mode,
        create_backup=not args.no_backup
    )
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()

