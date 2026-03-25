#!/usr/bin/env python3
"""
Direct launcher for Alzheimer's Disease Detection Web App
"""

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
        input("Press Enter to exit...")
        return
    
    print("✅ Model found!")
    print("🚀 Starting web app...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C in the terminal to stop the app")
    print("=" * 40)
    
    # Import and run Streamlit directly
    try:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "localhost"]
        stcli.main()
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure all packages are installed:")
        print("   pip install streamlit plotly tensorflow pillow numpy pandas matplotlib seaborn")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
