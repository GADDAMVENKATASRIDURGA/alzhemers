"""
Simple Python API for fast predictions - Import and use directly in your code
No dataset evaluation, no command line needed

CRITICAL MEDICAL SAFETY NOTE:
This tool is designed ONLY for brain imaging analysis (MRI, CT, PET scans).
It CANNOT diagnose Alzheimer's from normal photographs, faces, or selfies.
All inputs MUST be validated as legitimate brain scans before processing.
"""
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from mri_validation import validate_mri_scan

# ============================================
# MEDICAL SAFETY: BRAIN SCAN VALIDATION
# ============================================

def validate_brain_scan(image_path):
    """Validate image with strict shared MRI rules."""
    return validate_mri_scan(image_path)


class AlzheimerPredictor:
    """Fast Alzheimer disease predictor - loads model once, predicts efficiently"""
    
    def __init__(self, model_path='best_alzheimer_model.h5'):
        """Initialize predictor with trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = keras.models.load_model(model_path, compile=False)
        self.class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        print(f"✅ Model loaded: {model_path}")
    
    def predict_image(self, image_path, return_all_probs=False):
        """
        Predict single image
        
        Args:
            image_path: Path to image file
            return_all_probs: If True, return all class probabilities
        
        Returns:
            If return_all_probs=False: (predicted_class, confidence)
            If return_all_probs=True: (predicted_class, confidence, all_probs_dict)
        
        MEDICAL SAFETY: This method validates input is a brain scan.
        
        Example:
            >>> predictor = AlzheimerPredictor()
            >>> disease, conf = predictor.predict_image('brain_mri.jpg')
            >>> print(f"Prediction: {disease} ({conf*100:.1f}%)")
        """
        # CRITICAL: Validate image is a brain scan
        is_valid, msg = validate_brain_scan(image_path)
        if not is_valid:
            print(f"[VALIDATION ERROR] {msg}")
            print("[MEDICAL SAFETY] Prediction aborted. This tool only processes brain scans (MRI, CT, PET). Normal photos, portraits, or non-medical images are strictly blocked. Please upload a valid brain scan.")
            return None, None, None if return_all_probs else (None, None)
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Predict
        preds = self.model.predict(arr, verbose=0)
        probs = preds[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        predicted_class = self.class_names[idx]
        
        if return_all_probs:
            all_probs = {name: float(prob) for name, prob in zip(self.class_names, probs)}
            return predicted_class, confidence, all_probs
        else:
            return predicted_class, confidence
    
    def predict_folder(self, folder_path, verbose=True):
        """
        Predict all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            verbose: Print progress
        
        Returns:
            List of (filename, predicted_class, confidence) tuples
        
        Example:
            >>> predictor = AlzheimerPredictor()
            >>> results = predictor.predict_folder('patient_mris/')
            >>> for filename, disease, conf in results:
            ...     print(f"{filename}: {disease} ({conf*100:.1f}%)")
        """
        results = []
        files = sorted([f for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        if verbose:
            print(f"Processing {len(files)} images...\n")
        
        for i, fname in enumerate(files, 1):
            path = os.path.join(folder_path, fname)
            try:
                pred_class, conf = self.predict_image(path)
                results.append((fname, pred_class, conf))
                if verbose:
                    print(f"[{i}/{len(files)}] {fname:30} -> {pred_class:20} ({conf*100:5.1f}%)")
            except Exception as e:
                if verbose:
                    print(f"[{i}/{len(files)}] {fname:30} -> ERROR: {e}")
        
        return results


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == '__main__':
    print("=" * 70)
    print("AlzheimerPredictor - Simple Python API Examples")
    print("=" * 70)
    
    # Example 1: Single image prediction
    print("\n📌 EXAMPLE 1: Predict Single Image")
    print("-" * 70)
    print("""
    from simple_predict import AlzheimerPredictor
    
    # Create predictor (loads model once)
    predictor = AlzheimerPredictor('best_alzheimer_model.h5')
    
    # Predict single image
    disease, confidence = predictor.predict_image('patient_mri.jpg')
    print(f"Result: {disease} ({confidence*100:.1f}%)")
    
    # Get all probabilities
    disease, confidence, all_probs = predictor.predict_image('patient_mri.jpg', 
                                                             return_all_probs=True)
    print(f"Full results: {all_probs}")
    """)
    
    # Example 2: Batch prediction
    print("\n📌 EXAMPLE 2: Predict Batch of Images")
    print("-" * 70)
    print("""
    # Predict entire folder
    results = predictor.predict_folder('patient_folder/')
    
    for filename, disease, confidence in results:
        print(f"{filename}: {disease} ({confidence*100:.1f}%)")
    """)
    
    # Example 3: Integration in larger application
    print("\n📌 EXAMPLE 3: Integration in Your Application")
    print("-" * 70)
    print("""
    from simple_predict import AlzheimerPredictor
    
    class ClinicSystem:
        def __init__(self):
            self.predictor = AlzheimerPredictor()
        
        def process_patient_mri(self, mri_path, patient_id):
            disease, confidence = self.predictor.predict_image(mri_path)
            
            # Store results
            self.store_diagnosis(patient_id, disease, confidence)
            
            # Alert if high risk
            if disease in ['MildDemented', 'ModerateDemented']:
                self.alert_doctor(patient_id, disease)
            
            return disease, confidence
    
    # Usage
    clinic = ClinicSystem()
    disease, conf = clinic.process_patient_mri('patient123_scan.jpg', 'PAT-001')
    """)
    
    print("\n✅ Ready to use! Import and create AlzheimerPredictor() in your code!")
