"""
Real-World Example: Using the Alzheimer's Model in Production
Shows how to integrate fast predictions into real applications
"""

# ============================================
# EXAMPLE 1: Hospital Clinic System
# ============================================

from simple_predict import AlzheimerPredictor
from datetime import datetime
import json

class HospitalClinicSystem:
    """Example clinic management system using fast predictions"""
    
    def __init__(self, model_path='best_alzheimer_model.h5'):
        # Load model once at startup (not for each prediction!)
        self.predictor = AlzheimerPredictor(model_path)
        self.records = []
    
    def process_patient_scan(self, patient_id, mri_path):
        """Process single patient MRI scan"""
        print(f"\n📋 Processing Patient {patient_id}")
        print(f"📷 Scan: {mri_path}")
        
        # Fast prediction (2 seconds)
        disease, confidence, all_probs = self.predictor.predict_image(
            mri_path, 
            return_all_probs=True
        )
        
        # Create diagnosis record
        diagnosis = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'mri_file': mri_path,
            'predicted_class': disease,
            'confidence': float(confidence),
            'all_probabilities': all_probs,
            'risk_level': self._get_risk_level(disease)
        }
        
        # Store in records
        self.records.append(diagnosis)
        
        # Alert if high risk
        if disease in ['MildDemented', 'ModerateDemented']:
            self._alert_neurologist(patient_id, disease, confidence)
        
        return diagnosis
    
    def process_patient_batch(self, patient_id, mri_folder):
        """Process multiple MRI scans for one patient"""
        print(f"\n📋 Batch Processing Patient {patient_id}")
        print(f"📁 Folder: {mri_folder}")
        
        # Fast batch prediction (30-60 seconds for many images)
        results = self.predictor.predict_folder(mri_folder)
        
        diagnoses = []
        for filename, disease, confidence in results:
            diagnosis = {
                'patient_id': patient_id,
                'timestamp': datetime.now().isoformat(),
                'scan_file': filename,
                'prediction': disease,
                'confidence': float(confidence),
                'risk_level': self._get_risk_level(disease)
            }
            diagnoses.append(diagnosis)
            self.records.append(diagnosis)
        
        return diagnoses
    
    def _get_risk_level(self, disease):
        """Convert disease class to risk level"""
        risk_map = {
            'NonDemented': 'LOW',
            'VeryMildDemented': 'VERY_LOW',
            'MildDemented': 'MODERATE',
            'ModerateDemented': 'HIGH'
        }
        return risk_map.get(disease, 'UNKNOWN')
    
    def _alert_neurologist(self, patient_id, disease, confidence):
        """Alert neurologist for high-risk cases"""
        print(f"\n⚠️ ALERT: High-risk case detected!")
        print(f"   Patient: {patient_id}")
        print(f"   Diagnosis: {disease}")
        print(f"   Confidence: {confidence*100:.1f}%")
        print(f"   Action: Notifying neurologist...")
    
    def export_report(self, filename='patient_report.json'):
        """Export all diagnosis records"""
        with open(filename, 'w') as f:
            json.dump(self.records, f, indent=2)
        print(f"✅ Report saved: {filename}")


# ============================================
# EXAMPLE 2: Research Study Analysis
# ============================================

from collections import Counter
import statistics

class ResearchStudyAnalyzer:
    """Example for analyzing groups of patients in research"""
    
    def __init__(self, model_path='best_alzheimer_model.h5'):
        self.predictor = AlzheimerPredictor(model_path)
        self.study_results = []
    
    def analyze_study_group(self, group_name, mri_folder):
        """Analyze entire study group"""
        print(f"\n📊 Analyzing Study Group: {group_name}")
        
        # Batch prediction on all images
        results = self.predictor.predict_folder(mri_folder)
        
        # Extract results
        predictions = [disease for _, disease, _ in results]
        confidences = [conf for _, _, conf in results]
        
        # Calculate statistics
        counts = Counter(predictions)
        avg_confidence = statistics.mean(confidences)
        
        # Store results
        study_data = {
            'group': group_name,
            'total_scans': len(results),
            'predictions': dict(counts),
            'average_confidence': avg_confidence,
            'scan_details': results
        }
        
        self.study_results.append(study_data)
        
        # Display summary
        print(f"✅ Processed {len(results)} scans")
        print(f"   Average Confidence: {avg_confidence*100:.1f}%")
        for disease, count in counts.items():
            percentage = (count / len(results)) * 100
            print(f"   {disease}: {count} ({percentage:.1f}%)")
        
        return study_data


# ============================================
# EXAMPLE 3: Real-Time Monitoring System
# ============================================

from pathlib import Path
import time

class MRIMonitoringSystem:
    """Monitor a folder for new MRI scans and predict automatically"""
    
    def __init__(self, model_path='best_alzheimer_model.h5'):
        self.predictor = AlzheimerPredictor(model_path)
        self.processed_files = set()
    
    def monitor_and_predict(self, watch_folder, check_interval=30):
        """
        Watch a folder and automatically predict on new images
        
        Args:
            watch_folder: Folder path to monitor
            check_interval: Seconds between checks
        """
        print(f"👀 Monitoring folder: {watch_folder}")
        print(f"   Check interval: {check_interval}s")
        print("   Press Ctrl+C to stop")
        
        try:
            while True:
                # Find new images
                new_files = self._find_new_images(watch_folder)
                
                if new_files:
                    print(f"\n🆕 Found {len(new_files)} new image(s)")
                    
                    for filepath in new_files:
                        disease, conf = self.predictor.predict_image(str(filepath))
                        print(f"   ✓ {filepath.name}: {disease} ({conf*100:.1f}%)")
                        self.processed_files.add(str(filepath))
                
                # Wait before checking again
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped")
    
    def _find_new_images(self, folder):
        """Find image files not yet processed"""
        path = Path(folder)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        new_files = [
            f for f in path.iterdir()
            if f.suffix.lower() in extensions and str(f) not in self.processed_files
        ]
        
        return new_files


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == '__main__':
    print("=" * 80)
    print("ALZHEIMER'S MODEL - PRODUCTION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Hospital clinic
    print("\n\n📌 EXAMPLE 1: Hospital Clinic System")
    print("-" * 80)
    print("""
    # Initialize clinic system
    clinic = HospitalClinicSystem('best_alzheimer_model.h5')
    
    # Process single patient scan
    diagnosis = clinic.process_patient_scan(
        patient_id='PAT-001',
        mri_path='patient_scans/patient_001.jpg'
    )
    
    # Process multiple scans for patient
    diagnoses = clinic.process_patient_batch(
        patient_id='PAT-002',
        mri_folder='patient_scans/patient_002/'
    )
    
    # Export all records
    clinic.export_report('clinic_records.json')
    """)
    
    # Example 2: Research study
    print("\n\n📌 EXAMPLE 2: Research Study Analysis")
    print("-" * 80)
    print("""
    # Initialize research analyzer
    researcher = ResearchStudyAnalyzer('best_alzheimer_model.h5')
    
    # Analyze study groups
    control_group = researcher.analyze_study_group(
        'Control Group',
        'data/control/'
    )
    
    alzheimer_group = researcher.analyze_study_group(
        'Alzheimer Group',
        'data/alzheimer/'
    )
    """)
    
    # Example 3: Real-time monitoring
    print("\n\n📌 EXAMPLE 3: Real-Time Monitoring")
    print("-" * 80)
    print("""
    # Initialize monitoring system
    monitor = MRIMonitoringSystem('best_alzheimer_model.h5')
    
    # Monitor folder for new MRI scans
    monitor.monitor_and_predict(
        watch_folder='new_mri_scans/',
        check_interval=30  # Check every 30 seconds
    )
    """)
    
    print("\n\n✅ All examples ready to use in your production code!")
    print("\nKey Benefits:")
    print("  ⚡ Fast: Single prediction in ~2 seconds")
    print("  🎯 Accurate: Same model, same high accuracy")
    print("  🔄 Efficient: Load model once, predict many times")
    print("  🚀 Production-Ready: Easy integration in real systems")
