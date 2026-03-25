"""
Lightweight prediction script - NO dataset evaluation, only processes user-provided images
Fast and efficient for single image or batch predictions

CRITICAL MEDICAL SAFETY NOTE:
This tool is designed ONLY for brain MRI imaging analysis.
It CANNOT diagnose Alzheimer's from normal photographs, faces, or selfies.
All inputs MUST be validated as legitimate brain MRI scans before processing.
"""
import os
import sys
import numpy as np
from PIL import Image
from tensorflow import keras
from mri_validation import validate_mri_scan as strict_validate_mri_scan

# ============================================
# MEDICAL SAFETY: MRI SCAN VALIDATION
# ============================================

def validate_mri_scan(image_path):
    """Validate image with strict shared MRI rules."""
    return strict_validate_mri_scan(image_path)

# ============================================
# PREDICTION HELPERS (FAST - NO DATASET LOADING)
# ============================================

def _get_class_names_fallback():
    """Get class names for the model"""
    return ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def load_trained_model(model_path='best_alzheimer_model.h5'):
    """Load a trained Keras model. Returns the model or raises an error."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        loaded = keras.models.load_model(model_path, compile=False)
        print(f"[OK] Loaded model: {model_path}")
        return loaded
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image file, convert to RGB, resize, scale to [0,1], and return a batch tensor.
    Fast preprocessing without any augmentation.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    return arr

def predict_image(model, image_path, class_names=None, target_size=(224, 224)):
    """
    Predict single image and return (label, confidence, all_probabilities).
    FAST - No dataset evaluation!
    
    CRITICAL: This function should ONLY be used with legitimate brain MRI scans.
    """
    if class_names is None:
        class_names = _get_class_names_fallback()
    
    # Validate image is an MRI scan
    is_valid, validation_msg = validate_mri_scan(image_path)
    if not is_valid:
        print(f"\n[VALIDATION ERROR] {validation_msg}")
        return None, None, None
    
    x = preprocess_image(image_path, target_size=target_size)
    preds = model.predict(x, verbose=0)  # Silent prediction
    probs = preds[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else str(idx)
    return label, float(probs[idx]), probs

def predict_folder(model, folder_path, class_names=None, exts=('.jpg', '.jpeg', '.png'), target_size=(224, 224)):
    """
    Run predictions on all image files in a folder.
    Returns list of (path, label, confidence).
    FAST - Process all images without dataset evaluation!
    """
    results = []
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])
    if not files:
        print(f"[!] No image files found in {folder_path}")
        return results
    
    print(f"Processing {len(files)} images...\n")
    
    for i, fname in enumerate(files, 1):
        path = os.path.join(folder_path, fname)
        try:
            label, prob, _ = predict_image(model, path, class_names=class_names, target_size=target_size)
            if label is None:
                print(f"[{i}/{len(files)}] {fname:30} -> [SKIPPED]: validation failed")
                continue
            results.append((path, label, prob))
            print(f"[{i}/{len(files)}] {fname:30} -> {label:20} ({prob*100:5.1f}%)")
        except Exception as e:
            print(f"[{i}/{len(files)}] {fname:30} -> [ERROR]: {e}")
    
    return results

# ============================================
# CLI INTERFACE
# ============================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FAST MRI Prediction - Predict Alzheimer disease on user-provided images (NO dataset evaluation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single image
  python predict.py -i path/to/image.jpg
  
  # Predict folder of images
  python predict.py -i path/to/folder/
  
  # Use custom model
  python predict.py -i image.jpg -m my_model.h5
  
  # Custom image size
  python predict.py -i image.jpg -s 224 224
        """
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True, 
        help='Path to image file or folder containing images'
    )
    parser.add_argument(
        '--model', '-m', 
        default='best_alzheimer_model.h5', 
        help='Path to trained model (.h5) [default: best_alzheimer_model.h5]'
    )
    parser.add_argument(
        '--size', '-s',
        type=int, 
        nargs=2, 
        metavar=('W', 'H'), 
        default=(224, 224), 
        help='Target image size W H [default: 224 224]'
    )
    
    args = parser.parse_args()
    
    # Load model (FAST - no dataset loading!)
    print("[*] Loading model...")
    mdl = load_trained_model(args.model)
    classes = _get_class_names_fallback()
    
    print(f"[*] Using classes: {', '.join(classes)}\n")
    
    # Process input
    if os.path.isdir(args.input):
        # Batch prediction
        results = predict_folder(
            mdl,
            args.input,
            class_names=classes,
            exts=('.jpg', '.jpeg', '.png', '.bmp'),
            target_size=tuple(args.size),
        )
        
        if results:
            print(f"\n{'='*70}")
            print(f"[*] SUMMARY: {len(results)} images processed")
            print(f"{'='*70}")
            
            # Count predictions
            from collections import Counter
            predictions = [r[1] for r in results]
            counts = Counter(predictions)
            
            for class_name in classes:
                count = counts.get(class_name, 0)
                print(f"  {class_name:20} : {count}")
    
    elif os.path.isfile(args.input):
        # Single image prediction
        print(f"Analyzing: {args.input}\n")
        lbl, pr, probs = predict_image(mdl, args.input, class_names=classes, target_size=tuple(args.size))
        
        if lbl is None:  # Validation failed
            print("\n" + "="*70)
            print("[VALIDATION FAILED - MEDICAL SAFETY CHECK]")
            print("="*70)
            print("This tool analyzes ONLY brain MRI imaging data.")
            print("It CANNOT diagnose Alzheimer's from normal photographs or selfies.")
            print("\nFor medical diagnosis, please:")
            print("  1. Obtain proper brain MRI imaging from a medical facility")
            print("  2. Have a radiologist review the scans")
            print("  3. Consult with a qualified neurologist")
            sys.exit(1)
        
        print(f"{'='*70}")
        print(f"[*] PREDICTION RESULT")
        print(f"{'='*70}")
        print(f"Classification: {lbl}")
        print(f"Confidence:     {pr*100:.2f}%")
        print(f"\n[*] ALL PROBABILITIES:")
        print(f"{'-'*70}")
        for i, (class_name, prob) in enumerate(zip(classes, probs)):
            bar = '#' * int(prob * 30) + '-' * int((1-prob) * 30)
            print(f"  {class_name:20} : {bar} {prob*100:5.1f}%")
        print(f"{'='*70}")
        
        # Medical disclaimer
        print("\n" + "="*70)
        print("[MEDICAL DISCLAIMER]")
        print("="*70)
        print("This is an AI model output for RESEARCH PURPOSES ONLY.")
        print("DO NOT use for clinical diagnosis without professional review.")
        print("Always consult qualified healthcare professionals.")
        print("="*70)
    
    else:
        print(f"[ERROR] Input path not found: {args.input}")
        sys.exit(1)
    
    print("\n[OK] Done! (Fast - no dataset evaluation needed)")
