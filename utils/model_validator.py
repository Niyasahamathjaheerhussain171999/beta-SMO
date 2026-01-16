"""
ScoutMe - Model Validator Utility
Validates AI models before processing to catch configuration errors early.

USAGE:
    python utils/model_validator.py config.yaml

VALIDATES:
- Model files exist
- Model architecture matches expected task
- Class names are correct (not generic COCO classes)
- Models can be loaded successfully
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


class ModelValidator:
    """
    Validates YOLO models match expected configurations.
    Catches wrong model loading (e.g., COCO model instead of custom football model).
    """
    
    # Expected configurations for each model type
    EXPECTED_CONFIGS = {
        'player_detector': {
            'expected_classes': ['player', 'goalkeeper', 'referee', 'ball', 'person'],
            'forbidden_classes': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'],
            'min_classes': 1,
            'description': 'Football player detection model'
        },
        'ball_detector': {
            'expected_classes': ['ball', 'sports ball', 'football', 'soccer ball'],
            'forbidden_classes': ['bicycle', 'car', 'motorcycle', 'airplane'],
            'min_classes': 1,
            'description': 'Football/soccer ball detection model'
        },
        'pitch_detector': {
            'expected_classes': [
                'corner', 'penalty_spot', 'center_circle', 'goal_post',
                'penalty_box', 'center_line', 'touchline', 'goal_line',
                'pitch', 'field', 'line', 'keypoint'
            ],
            'forbidden_classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus'],
            'min_classes': 1,
            'description': 'Football pitch/field keypoint detection model'
        }
    }
    
    # COCO dataset classes (to detect wrong model)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella'
    ]
    
    def __init__(self):
        self.validation_results = {}
        self.all_valid = True
    
    def validate_model(self, model_path: str, model_type: str) -> Dict:
        """
        Validate a single YOLO model.
        
        Args:
            model_path: Path to .pt model file
            model_type: One of 'player_detector', 'ball_detector', 'pitch_detector'
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': False,
            'model_path': model_path,
            'model_type': model_type,
            'error': None,
            'warnings': [],
            'detected_classes': [],
            'fix': None
        }
        
        # Check if path exists
        if not Path(model_path).exists():
            result['error'] = f"Model file not found: {model_path}"
            result['fix'] = f"Download or copy the correct {model_type} model to: {model_path}"
            return result
        
        # Try to load model
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            class_names = model.names  # Dict: {0: 'person', 1: 'bicycle', ...}
            result['detected_classes'] = list(class_names.values())
        except ImportError:
            result['error'] = "ultralytics package not installed"
            result['fix'] = "pip install ultralytics"
            return result
        except Exception as e:
            result['error'] = f"Failed to load model: {str(e)}"
            result['fix'] = "Model file may be corrupted - re-download"
            return result
        
        # Get expected config
        if model_type not in self.EXPECTED_CONFIGS:
            result['warnings'].append(f"Unknown model type: {model_type}")
            result['valid'] = True  # Allow unknown types
            return result
        
        config = self.EXPECTED_CONFIGS[model_type]
        
        # Check for forbidden classes (indicates wrong model)
        forbidden_found = [cls for cls in config['forbidden_classes'] 
                         if cls in class_names.values()]
        
        # Special case: player detector can have 'person' class
        if model_type == 'player_detector':
            forbidden_found = [cls for cls in forbidden_found if cls != 'person']
        
        if len(forbidden_found) >= 3:  # Multiple COCO classes = wrong model
            result['error'] = f"WRONG MODEL! This appears to be a generic COCO model, not a {config['description']}"
            result['fix'] = f"Replace with a custom-trained football {model_type} model"
            result['detected_classes'] = list(class_names.values())[:10]
            return result
        
        # Check for expected classes
        expected_found = [cls for cls in config['expected_classes'] 
                        if any(cls.lower() in str(v).lower() for v in class_names.values())]
        
        # For pitch detector, be more lenient - any keypoint-related class is OK
        if model_type == 'pitch_detector':
            # Check if it has pitch-related classes or keypoint indices
            has_pitch_classes = len(expected_found) > 0
            has_numbered_keypoints = all(isinstance(k, int) for k in class_names.keys())
            
            if not has_pitch_classes and not has_numbered_keypoints:
                # Check if all classes are COCO classes
                coco_count = sum(1 for v in class_names.values() if v in self.COCO_CLASSES)
                if coco_count > len(class_names) * 0.5:
                    result['error'] = f"WRONG MODEL! Loaded a COCO-trained model instead of pitch keypoint detector"
                    result['fix'] = (
                        "You need a pitch keypoint detection model, not a generic object detector.\n"
                        "      Options:\n"
                        "      1. Train a custom pitch keypoint model on Roboflow\n"
                        "      2. Use a pre-trained football pitch model\n"
                        "      3. Disable pitch detection and use fallback distance estimation"
                    )
                    return result
        
        # Model passed validation
        result['valid'] = True
        
        # Add warnings for potential issues
        if len(result['detected_classes']) > 50:
            result['warnings'].append("Model has many classes - may be slower than specialized model")
        
        return result
    
    def validate_all_models(self, config: Dict) -> Dict[str, Dict]:
        """
        Validate all models defined in config.
        
        Args:
            config: Configuration dictionary with 'models' section
            
        Returns:
            Dictionary of validation results per model
        """
        results = {}
        
        models_config = config.get('models', {})
        
        # Map config keys to model types
        model_mappings = {
            'player_detection': 'player_detector',
            'ball_detection': 'ball_detector',
            'pitch_detection': 'pitch_detector',
        }
        
        for config_key, model_type in model_mappings.items():
            if config_key in models_config:
                model_path = models_config[config_key]
                
                # Handle relative paths
                if not os.path.isabs(model_path):
                    # Try common locations
                    possible_paths = [
                        model_path,
                        f"data/input/{os.path.basename(model_path)}",
                        f"models/{os.path.basename(model_path)}",
                        f"data/{os.path.basename(model_path)}",
                    ]
                    
                    found_path = None
                    for p in possible_paths:
                        if Path(p).exists():
                            found_path = p
                            break
                    
                    if found_path:
                        model_path = found_path
                
                results[config_key] = self.validate_model(model_path, model_type)
                
                if not results[config_key]['valid']:
                    self.all_valid = False
        
        self.validation_results = results
        return results
    
    def print_validation_report(self, results: Optional[Dict] = None) -> bool:
        """
        Print formatted validation report.
        
        Returns:
            True if all models valid, False otherwise
        """
        if results is None:
            results = self.validation_results
        
        print(f"\n{'='*80}")
        print("🤖 MODEL VALIDATION REPORT")
        print(f"{'='*80}")
        
        all_valid = True
        
        for model_name, result in results.items():
            print(f"\n📦 {model_name}:")
            print(f"   Path: {result['model_path']}")
            
            if result['valid']:
                print(f"   Status: ✅ Valid")
                if result['detected_classes']:
                    classes_preview = ', '.join(result['detected_classes'][:5])
                    if len(result['detected_classes']) > 5:
                        classes_preview += f"... (+{len(result['detected_classes'])-5} more)"
                    print(f"   Classes: {classes_preview}")
            else:
                all_valid = False
                print(f"   Status: ❌ INVALID")
                print(f"   Error: {result['error']}")
                if result['detected_classes']:
                    classes_preview = ', '.join(result['detected_classes'][:5])
                    print(f"   Detected classes: {classes_preview}...")
                if result['fix']:
                    print(f"   💡 Fix: {result['fix']}")
            
            for warning in result.get('warnings', []):
                print(f"   ⚠️ Warning: {warning}")
        
        print(f"\n{'='*80}")
        if all_valid:
            print("✅ ALL MODELS VALIDATED SUCCESSFULLY")
        else:
            print("❌ MODEL VALIDATION FAILED - Fix issues above before processing")
        print(f"{'='*80}\n")
        
        return all_valid


def validate_models_from_config(config_path: str = 'config.yaml') -> bool:
    """
    Quick function to validate models from config file.
    
    Returns:
        True if all models valid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    validator = ModelValidator()
    results = validator.validate_all_models(config)
    return validator.print_validation_report(results)


def main():
    """CLI entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    
    print(f"📋 Loading config from: {config_path}")
    
    valid = validate_models_from_config(config_path)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()

