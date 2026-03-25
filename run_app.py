#!/usr/bin/env python3
"""
Launcher script for Alzheimer's Disease Detection Web App
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import plotly
        import tensorflow
        from PIL import Image  # Test PIL import specifically
        import numpy
        import pandas
        import matplotlib
        import seaborn
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n💡 Install with: pip install -r requirements_ui.txt")
        return False

def check_model():
    """Check if model file exists"""
    model_path = "best_alzheimer_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure the trained model is in the same directory")
        return False
    return True

def main():
    print("🧠 Alzheimer's Disease Detection App Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check model
    if not check_model():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🚀 Starting web app...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("=" * 50)
    
    # Run Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
