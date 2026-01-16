#!/usr/bin/env python3
"""
ScoutMe - Pre-Flight Checklist & Analysis Runner
Runs comprehensive checks before video analysis to catch issues early.

USAGE:
    python run_analysis_with_checks.py path/to/video.mp4
    python run_analysis_with_checks.py path/to/video.mp4 --skip-checks
    python run_analysis_with_checks.py path/to/video.mp4 --enhance-video

CHECKS PERFORMED:
1. Video Health Check (corruption, truncation)
2. Model Validation (correct models loaded)
3. GPU Availability
4. Output Directory Setup
5. Configuration Validation
"""

import sys
import os
import argparse
from pathlib import Path
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_preflight_checks(video_path: str, config_path: str = 'config.yaml') -> tuple:
    """
    Run all pre-flight checks before analysis.
    
    Args:
        video_path: Path to video file
        config_path: Path to configuration file
        
    Returns:
        (passed: bool, processed_video_path: str or None)
    """
    print(f"\n{'='*80}")
    print("🚀 SCOUTME PRE-FLIGHT CHECKLIST")
    print(f"{'='*80}\n")
    
    all_passed = True
    processed_video = video_path
    
    # =========================================================================
    # CHECK 1: Video Health
    # =========================================================================
    print("1️⃣  CHECKING VIDEO HEALTH...")
    print("-" * 60)
    
    try:
        from utils.video_health_check import VideoHealthChecker
        
        checker = VideoHealthChecker(video_path)
        report = checker.run_full_diagnostic()
        checker.generate_report()
        
        if report['has_critical_issues']:
            print("   ❌ Video has critical issues")
            all_passed = False
            
            if report['is_truncated']:
                print("\n   💡 ATTEMPTING AUTO-REPAIR...")
                try:
                    from utils.video_processor import VideoProcessor
                    processor = VideoProcessor(video_path)
                    repaired = processor.process(mode='repair', create_backup=True)
                    
                    if repaired:
                        print(f"   ✅ Video repaired: {repaired}")
                        processed_video = str(repaired)
                        all_passed = True  # Continue with repaired video
                    else:
                        print("   ❌ Auto-repair failed")
                except Exception as e:
                    print(f"   ❌ Auto-repair error: {e}")
        else:
            print("   ✅ Video is healthy")
            
    except ImportError as e:
        print(f"   ⚠️ Could not import video health check: {e}")
        print("   Continuing without video health check...")
    except Exception as e:
        print(f"   ⚠️ Video health check failed: {e}")
        print("   Continuing anyway...")
    
    # =========================================================================
    # CHECK 2: Model Validation
    # =========================================================================
    print("\n2️⃣  VALIDATING AI MODELS...")
    print("-" * 60)
    
    try:
        # Load config
        config = {}
        if Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        
        from utils.model_validator import ModelValidator
        
        validator = ModelValidator()
        model_results = validator.validate_all_models(config)
        models_valid = validator.print_validation_report(model_results)
        
        if not models_valid:
            print("   ❌ Model validation failed")
            # Don't fail completely - models might still work
            print("   ⚠️ Continuing with warnings...")
        else:
            print("   ✅ All models validated")
            
    except ImportError as e:
        print(f"   ⚠️ Could not import model validator: {e}")
        print("   Continuing without model validation...")
    except Exception as e:
        print(f"   ⚠️ Model validation failed: {e}")
        print("   Continuing anyway...")
    
    # =========================================================================
    # CHECK 3: GPU Availability
    # =========================================================================
    print("\n3️⃣  CHECKING GPU...")
    print("-" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU available: {gpu_name}")
            print(f"      Memory: {gpu_memory:.1f} GB")
            
            # Check available memory
            free_memory = torch.cuda.mem_get_info()[0] / 1e9
            print(f"      Free: {free_memory:.1f} GB")
            
            if free_memory < 4:
                print("   ⚠️ Low GPU memory - may affect performance")
        else:
            print("   ⚠️ No GPU detected - will use CPU (SLOW!)")
            print("   💡 For best performance, run on a GPU machine")
            
    except ImportError:
        print("   ⚠️ PyTorch not installed - cannot check GPU")
    except Exception as e:
        print(f"   ⚠️ GPU check failed: {e}")
    
    # =========================================================================
    # CHECK 4: Output Directory
    # =========================================================================
    print("\n4️⃣  CHECKING OUTPUT DIRECTORY...")
    print("-" * 60)
    
    try:
        output_dir = Path(config.get('pipeline', {}).get('output_dir', 'data/output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Output directory ready: {output_dir}")
        
        # Check write permissions
        test_file = output_dir / '.write_test'
        try:
            test_file.write_text('test')
            test_file.unlink()
            print("   ✅ Write permissions OK")
        except Exception:
            print("   ❌ Cannot write to output directory!")
            all_passed = False
            
    except Exception as e:
        print(f"   ⚠️ Output directory check failed: {e}")
    
    # =========================================================================
    # CHECK 5: Configuration
    # =========================================================================
    print("\n5️⃣  VALIDATING CONFIGURATION...")
    print("-" * 60)
    
    try:
        if not config:
            print("   ⚠️ No configuration loaded - using defaults")
        else:
            # Check critical settings
            vlm_enabled = config.get('vlm', {}).get('enabled', True)
            print(f"   VLM Enabled: {'✅ Yes' if vlm_enabled else '❌ No'}")
            
            auto_shutdown = config.get('pipeline', {}).get('auto_shutdown_enabled', False)
            print(f"   Auto-Shutdown: {'✅ Yes' if auto_shutdown else '❌ No'}")
            
            confidence = config.get('detection', {}).get('confidence_threshold', 75)
            print(f"   Confidence Threshold: {confidence}%")
            
            print("   ✅ Configuration loaded")
            
    except Exception as e:
        print(f"   ⚠️ Configuration check failed: {e}")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ ALL PRE-FLIGHT CHECKS PASSED")
        print(f"   Video: {processed_video}")
        print("   Ready to process!")
    else:
        print("❌ PRE-FLIGHT CHECKS FAILED")
        print("   Fix the issues above before processing")
    print(f"{'='*80}\n")
    
    return all_passed, processed_video if all_passed else None


def run_analysis(video_path: str, enhance: bool = False):
    """
    Run the main analysis pipeline.
    
    Args:
        video_path: Path to video file
        enhance: Whether to enhance video before processing
    """
    print(f"\n{'='*80}")
    print("🎬 STARTING ANALYSIS")
    print(f"{'='*80}\n")
    
    # Optionally enhance video
    if enhance:
        print("🎨 Enhancing video for better AI detection...")
        try:
            from utils.video_processor import VideoProcessor
            processor = VideoProcessor(video_path)
            enhanced = processor.process(mode='enhance', create_backup=True)
            if enhanced:
                video_path = str(enhanced)
                print(f"   ✅ Using enhanced video: {video_path}")
            else:
                print("   ⚠️ Enhancement failed - using original video")
        except Exception as e:
            print(f"   ⚠️ Enhancement error: {e}")
            print("   Continuing with original video...")
    
    # Import and run main analysis
    try:
        # Try to import main4 module
        import main4
        
        # Check if main4 has a run function or needs to be called differently
        if hasattr(main4, 'main'):
            # If there's a main function, we need to set up sys.argv
            original_argv = sys.argv
            sys.argv = ['main4.py', '--source', video_path]
            main4.main()
            sys.argv = original_argv
        elif hasattr(main4, 'run_analysis'):
            main4.run_analysis(video_path)
        else:
            # Run as script
            print(f"   Running: python main4.py --source {video_path}")
            import subprocess
            result = subprocess.run(
                [sys.executable, 'main4.py', '--source', video_path],
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode != 0:
                print(f"   ❌ Analysis failed with exit code {result.returncode}")
                return False
                
    except ImportError as e:
        print(f"   ❌ Could not import main4: {e}")
        print("   Make sure main4.py exists in the project root")
        return False
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ScoutMe Analysis with Pre-Flight Checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis_with_checks.py data/input/match.mp4
  python run_analysis_with_checks.py data/input/match.mp4 --skip-checks
  python run_analysis_with_checks.py data/input/match.mp4 --enhance-video
  python run_analysis_with_checks.py data/input/match.mp4 --checks-only
        """
    )
    
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip pre-flight checks (not recommended)')
    parser.add_argument('--enhance-video', action='store_true',
                       help='Enhance video quality before processing (slow)')
    parser.add_argument('--checks-only', action='store_true',
                       help='Only run checks, do not process video')
    parser.add_argument('--repair', action='store_true',
                       help='Attempt to repair video if corrupted')
    
    args = parser.parse_args()
    
    # Validate video path exists
    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Run pre-flight checks
    if args.skip_checks:
        print("⚠️ Skipping pre-flight checks (not recommended)")
        processed_video = args.video_path
        passed = True
    else:
        passed, processed_video = run_preflight_checks(args.video_path, args.config)
    
    # Handle repair-only mode
    if args.repair and not passed:
        print("\n💡 Attempting video repair...")
        try:
            from utils.video_processor import VideoProcessor
            processor = VideoProcessor(args.video_path)
            repaired = processor.process(mode='repair', create_backup=True)
            if repaired:
                print(f"✅ Video repaired: {repaired}")
                processed_video = str(repaired)
                passed = True
        except Exception as e:
            print(f"❌ Repair failed: {e}")
    
    # Exit if checks failed
    if not passed:
        print("\n❌ Cannot proceed - fix issues above first")
        sys.exit(1)
    
    # Exit if checks-only mode
    if args.checks_only:
        print("\n✅ Pre-flight checks complete (--checks-only mode)")
        sys.exit(0)
    
    # Run analysis
    success = run_analysis(processed_video, enhance=args.enhance_video)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

