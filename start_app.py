#!/usr/bin/env python3
"""
Simple launcher for Alzheimer's Disease Detection Web App
"""

import subprocess
import sys
import os

def main():
    print("🧠 Alzheimer's Disease Detection App")
    print("=" * 40)
    
    # Check if model exists
    model_path = "best_alzheimer_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure the trained model is in the same directory")
        return
    
    print("✅ Model found!")
    print("🚀 Starting web app...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("=" * 40)
    
    # Run Streamlit app directly
    try:
        # Use the current Python interpreter (from virtual environment)
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        print("\n💡 Make sure all packages are installed:")
        print("   pip install streamlit plotly tensorflow pillow numpy pandas matplotlib seaborn")

if __name__ == "__main__":
    main()
