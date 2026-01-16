"""
ScoutMe - Video Health Check Utility
Comprehensive diagnostic tool for video files before AI processing.

USAGE:
    python utils/video_health_check.py path/to/video.mp4

DETECTS:
- File corruption (NAL unit errors, truncation)
- Missing/corrupted frames
- moov atom position (affects streaming)
- Codec compatibility
- Resolution/FPS verification
"""

import cv2
import subprocess
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import struct


class VideoHealthChecker:
    """
    Comprehensive video health diagnostic tool.
    Checks for corruption, truncation, and compatibility issues.
    """
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.issues = []
        self.warnings = []
        self.metadata = {}
        self.has_critical_issues = False
        self.is_truncated = False
        
    def check_file_exists(self) -> bool:
        """Basic file existence and readability check."""
        if not self.video_path.exists():
            self.issues.append({
                "type": "CRITICAL",
                "message": f"File not found: {self.video_path}",
                "fix": "Check the file path or re-download the video"
            })
            self.has_critical_issues = True
            return False
        
        if not self.video_path.is_file():
            self.issues.append({
                "type": "CRITICAL",
                "message": f"Path is not a file: {self.video_path}",
                "fix": "Provide a valid video file path"
            })
            self.has_critical_issues = True
            return False
        
        # Check file size
        file_size = self.video_path.stat().st_size
        self.metadata["file_size_bytes"] = file_size
        self.metadata["file_size_mb"] = file_size / (1024 * 1024)
        
        if file_size < 1000:  # Less than 1KB
            self.issues.append({
                "type": "CRITICAL",
                "message": f"File too small ({file_size} bytes) - likely empty or corrupted",
                "fix": "Re-download the video file"
            })
            self.has_critical_issues = True
            return False
        
        return True
    
    def check_ffprobe_metadata(self) -> bool:
        """Use ffprobe to extract deep metadata and detect issues."""
        try:
            # Check if ffprobe is available
            result = subprocess.run(
                ['ffprobe', '-version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.warnings.append({
                    "type": "WARNING",
                    "message": "ffprobe not available - skipping deep analysis",
                    "fix": "Install ffmpeg for comprehensive video analysis"
                })
                return True
        except FileNotFoundError:
            self.warnings.append({
                "type": "WARNING",
                "message": "ffprobe not found - skipping deep analysis",
                "fix": "Install ffmpeg: apt-get install ffmpeg (Linux) or download from ffmpeg.org"
            })
            return True
        
        # Run ffprobe with JSON output
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_format',
            '-show_streams',
            '-print_format', 'json',
            str(self.video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                if "Invalid NAL unit size" in error_msg or "partial file" in error_msg:
                    self.issues.append({
                        "type": "CRITICAL",
                        "message": "Video file is corrupted (NAL unit errors or truncation)",
                        "fix": "Re-download the video or use ffmpeg to repair: ffmpeg -i input.mp4 -c copy output.mp4"
                    })
                    self.has_critical_issues = True
                    self.is_truncated = True
                else:
                    self.warnings.append({
                        "type": "WARNING",
                        "message": f"ffprobe error: {error_msg[:200]}",
                        "fix": "Video may have minor issues but could still be processable"
                    })
                return False
            
            # Parse JSON output
            probe_data = json.loads(result.stdout)
            
            # Extract format info
            if "format" in probe_data:
                fmt = probe_data["format"]
                self.metadata["format"] = fmt.get("format_name", "unknown")
                self.metadata["duration_sec"] = float(fmt.get("duration", 0))
                self.metadata["duration_formatted"] = self._format_duration(self.metadata["duration_sec"])
                self.metadata["bitrate"] = int(fmt.get("bit_rate", 0))
                self.metadata["bitrate_mbps"] = self.metadata["bitrate"] / 1_000_000
            
            # Extract video stream info
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    self.metadata["codec"] = stream.get("codec_name", "unknown")
                    self.metadata["width"] = stream.get("width", 0)
                    self.metadata["height"] = stream.get("height", 0)
                    self.metadata["resolution"] = f"{self.metadata['width']}x{self.metadata['height']}"
                    
                    # Parse FPS
                    fps_str = stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        self.metadata["fps"] = float(num) / float(den) if float(den) > 0 else 0
                    else:
                        self.metadata["fps"] = float(fps_str)
                    
                    # Calculate expected frames
                    self.metadata["expected_frames"] = int(self.metadata["duration_sec"] * self.metadata["fps"])
                    
                    # Check for known issues
                    if self.metadata["codec"] not in ["h264", "hevc", "h265", "vp9", "av1"]:
                        self.warnings.append({
                            "type": "WARNING",
                            "message": f"Unusual codec: {self.metadata['codec']}",
                            "fix": "Consider re-encoding to H.264 for best compatibility"
                        })
                    break
            
            return True
            
        except subprocess.TimeoutExpired:
            self.issues.append({
                "type": "CRITICAL",
                "message": "ffprobe timed out - video may be severely corrupted",
                "fix": "Re-download the video file"
            })
            self.has_critical_issues = True
            return False
        except json.JSONDecodeError:
            self.warnings.append({
                "type": "WARNING",
                "message": "Could not parse ffprobe output",
                "fix": "Video metadata may be malformed"
            })
            return True
        except Exception as e:
            self.warnings.append({
                "type": "WARNING",
                "message": f"ffprobe analysis failed: {str(e)}",
                "fix": "Continuing with OpenCV analysis"
            })
            return True
    
    def check_opencv_readable(self) -> bool:
        """Verify OpenCV can read the video."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            self.issues.append({
                "type": "CRITICAL",
                "message": "OpenCV cannot open video file",
                "fix": "Video format may be incompatible - try re-encoding with ffmpeg"
            })
            self.has_critical_issues = True
            return False
        
        # Get OpenCV metadata
        self.metadata["opencv_fps"] = cap.get(cv2.CAP_PROP_FPS)
        self.metadata["opencv_frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.metadata["opencv_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.metadata["opencv_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try to read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            self.issues.append({
                "type": "CRITICAL",
                "message": "Cannot read first frame - video may be corrupted",
                "fix": "Re-download or repair video with ffmpeg"
            })
            self.has_critical_issues = True
            cap.release()
            return False
        
        cap.release()
        return True
    
    def check_frame_integrity(self, sample_count: int = 20) -> bool:
        """Sample frames throughout video to detect corruption."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            self.issues.append({
                "type": "CRITICAL",
                "message": "Video reports 0 frames",
                "fix": "Video file is likely corrupted"
            })
            self.has_critical_issues = True
            cap.release()
            return False
        
        # Sample frames at regular intervals
        step = max(1, total_frames // sample_count)
        frames_read = 0
        frames_failed = 0
        last_successful_frame = 0
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frames_read += 1
                last_successful_frame = i
            else:
                frames_failed += 1
        
        cap.release()
        
        self.metadata["frames_sampled"] = frames_read + frames_failed
        self.metadata["frames_readable"] = frames_read
        self.metadata["frames_failed"] = frames_failed
        self.metadata["last_readable_frame"] = last_successful_frame
        
        # Calculate actual readable percentage
        if total_frames > 0:
            readable_percent = (last_successful_frame / total_frames) * 100
            self.metadata["readable_percent"] = readable_percent
            
            if readable_percent < 95:
                self.issues.append({
                    "type": "CRITICAL",
                    "message": f"Video is truncated - only {readable_percent:.1f}% readable ({last_successful_frame}/{total_frames} frames)",
                    "fix": "Re-download the video file (current file is incomplete)"
                })
                self.has_critical_issues = True
                self.is_truncated = True
                return False
            elif readable_percent < 100:
                self.warnings.append({
                    "type": "WARNING",
                    "message": f"Video may be slightly truncated ({readable_percent:.1f}% readable)",
                    "fix": "Last few seconds may be missing"
                })
        
        if frames_failed > frames_read * 0.1:  # More than 10% failed
            self.issues.append({
                "type": "CRITICAL",
                "message": f"High frame failure rate: {frames_failed}/{frames_read + frames_failed} samples failed",
                "fix": "Video has significant corruption - re-download or repair"
            })
            self.has_critical_issues = True
            return False
        
        return True
    
    def check_moov_atom(self) -> bool:
        """Check if moov atom is at the start (required for streaming/seeking)."""
        try:
            with open(self.video_path, 'rb') as f:
                # Read first 8 bytes to check for moov atom
                header = f.read(32)
                
                # MP4 files should start with ftyp
                if b'ftyp' not in header[:12]:
                    self.warnings.append({
                        "type": "WARNING",
                        "message": "File may not be a standard MP4 (no ftyp atom)",
                        "fix": "Consider re-muxing with ffmpeg"
                    })
                
                # Check for moov atom in first 1MB (should be near start for fast start)
                f.seek(0)
                first_mb = f.read(1024 * 1024)
                
                if b'moov' in first_mb:
                    self.metadata["moov_position"] = "start"
                else:
                    # Check at end of file
                    f.seek(-1024 * 1024, 2)  # Last 1MB
                    last_mb = f.read()
                    
                    if b'moov' in last_mb:
                        self.metadata["moov_position"] = "end"
                        self.warnings.append({
                            "type": "WARNING",
                            "message": "moov atom is at end of file (slow seeking)",
                            "fix": "Run: ffmpeg -i input.mp4 -c copy -movflags +faststart output.mp4"
                        })
                    else:
                        self.metadata["moov_position"] = "unknown"
            
            return True
            
        except Exception as e:
            self.warnings.append({
                "type": "WARNING",
                "message": f"Could not check moov atom: {str(e)}",
                "fix": "File structure analysis skipped"
            })
            return True
    
    def run_full_diagnostic(self) -> Dict:
        """Run all checks and generate comprehensive report."""
        print(f"\n{'='*80}")
        print("🔍 VIDEO HEALTH DIAGNOSTIC")
        print(f"{'='*80}")
        print(f"📁 Analyzing: {self.video_path}")
        
        # Run all checks
        checks = [
            ("File Existence", self.check_file_exists),
            ("FFprobe Metadata", self.check_ffprobe_metadata),
            ("OpenCV Readability", self.check_opencv_readable),
            ("Frame Integrity", self.check_frame_integrity),
            ("File Structure", self.check_moov_atom),
        ]
        
        for check_name, check_func in checks:
            print(f"   Checking {check_name}...", end=" ")
            try:
                result = check_func()
                print("✅" if result else "❌")
            except Exception as e:
                print(f"⚠️ ({str(e)[:50]})")
        
        return {
            "video_path": str(self.video_path),
            "metadata": self.metadata,
            "issues": self.issues,
            "warnings": self.warnings,
            "has_critical_issues": self.has_critical_issues,
            "is_truncated": self.is_truncated
        }
    
    def generate_report(self) -> None:
        """Print formatted diagnostic report."""
        print(f"\n{'='*80}")
        print("📊 VIDEO HEALTH REPORT")
        print(f"{'='*80}")
        
        # File info
        print(f"\n📁 File Information:")
        print(f"   Path: {self.video_path}")
        print(f"   Size: {self.metadata.get('file_size_mb', 0):.2f} MB")
        print(f"   Format: {self.metadata.get('format', 'unknown')}")
        
        # Video properties
        print(f"\n📹 Video Properties:")
        print(f"   Resolution: {self.metadata.get('resolution', 'unknown')}")
        print(f"   FPS: {self.metadata.get('fps', 0):.2f}")
        print(f"   Duration: {self.metadata.get('duration_formatted', 'unknown')}")
        print(f"   Codec: {self.metadata.get('codec', 'unknown')}")
        print(f"   Bitrate: {self.metadata.get('bitrate_mbps', 0):.2f} Mbps")
        
        # Frame analysis
        if "expected_frames" in self.metadata:
            print(f"\n🎞️ Frame Analysis:")
            print(f"   Expected Frames: {self.metadata.get('expected_frames', 0):,}")
            print(f"   OpenCV Frame Count: {self.metadata.get('opencv_frame_count', 0):,}")
            print(f"   Last Readable Frame: {self.metadata.get('last_readable_frame', 0):,}")
            
            if "readable_percent" in self.metadata:
                pct = self.metadata["readable_percent"]
                status = "✅" if pct >= 99 else ("⚠️" if pct >= 90 else "❌")
                print(f"   Readable: {pct:.1f}% {status}")
        
        # moov atom
        if "moov_position" in self.metadata:
            moov_status = "✅ (fast start)" if self.metadata["moov_position"] == "start" else "⚠️ (slow seeking)"
            print(f"\n📦 File Structure:")
            print(f"   moov atom: {self.metadata['moov_position']} {moov_status}")
        
        # Issues
        if self.issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   ❌ {issue['message']}")
                print(f"      💡 Fix: {issue['fix']}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️ {warning['message']}")
                print(f"      💡 Fix: {warning['fix']}")
        
        # Final verdict
        print(f"\n{'='*80}")
        if self.has_critical_issues:
            print("❌ VERDICT: Video has critical issues - fix before processing")
            print("\n💡 RECOMMENDED ACTIONS:")
            print("   1. Re-download the video from source")
            print("   2. If re-download not possible, try repair:")
            print("      ffmpeg -i input.mp4 -c copy -movflags +faststart output.mp4")
            print("   3. If repair fails, re-encode:")
            print("      ffmpeg -i input.mp4 -c:v libx264 -crf 18 -c:a copy output.mp4")
        else:
            print("✅ VERDICT: Video is healthy and ready for processing")
        print(f"{'='*80}\n")
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python video_health_check.py <video_path>")
        print("Example: python video_health_check.py data/input/match.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    checker = VideoHealthChecker(video_path)
    report = checker.run_full_diagnostic()
    checker.generate_report()
    
    # Exit with error code if critical issues
    sys.exit(1 if report["has_critical_issues"] else 0)


if __name__ == "__main__":
    main()

