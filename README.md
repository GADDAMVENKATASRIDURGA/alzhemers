# 🧠 Alzheimer's Disease Detection Web App

A modern, interactive web interface for detecting Alzheimer's disease from MRI brain scans using deep learning.

## 🌟 Features

- **Single Image Analysis**: Upload individual MRI images for instant analysis
- **Batch Processing**: Analyze multiple images at once with summary statistics
- **Interactive Visualizations**: Beautiful charts showing prediction probabilities
- **Real-time Results**: Get instant predictions with confidence scores
- **Responsive Design**: Works on desktop and mobile devices
- **Risk Assessment**: Clear recommendations based on prediction results

## 🚀 Quick Start

### Prerequisites

1. **Trained Model**: Make sure you have `best_alzheimer_model.h5` in the project directory
2. **Python 3.8+**: Ensure Python is installed on your system

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_ui.txt
   ```

2. **Run the App**:
   ```bash
   python run_app.py
   ```

   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

3. **Open Browser**: The app will automatically open at `http://localhost:8501`

## 📸 How to Use

- **For Fast Command-Line Predictions:**
  - Use `predict.py` for single or batch image predictions:
    ```bash
    python predict.py -i "path/to/image.jpg"
    python predict.py -i "path/to/folder/"
    ```
- **For Python Integration:**
  - Use `simple_predict.py`:
    ```python
    from simple_predict import AlzheimerPredictor
    predictor = AlzheimerPredictor()
    disease, confidence = predictor.predict_image('patient_mri.jpg')
    print(f"Result: {disease} ({confidence*100:.1f}%)")
    ```

## 🗂 Project Structure

```
project_guntur/
├── app.py                   # Streamlit web app
├── run_app.py               # Launcher script
├── predict.py               # Fast CLI predictions
├── simple_predict.py        # Python API for predictions
├── best_alzheimer_model.h5  # Trained model (large file)
├── requirements_ui.txt      # Python dependencies
├── README.md                # This file
└── ... (other docs, scripts)
```

## ⚠️ Notes

- The model file (`best_alzheimer_model.h5`) is large (>50MB). For GitHub, consider using [Git LFS](https://git-lfs.github.com/) for version control of large files.
- For any issues, check the terminal output or open an issue on the repository.

## 🌐 Localhost Details

- By default, the web app runs at: [http://localhost:8501](http://localhost:8501)
- If you use `run_app.py`, it will launch the app and open your browser automatically.
- You can change the port by running:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

For more details, see the documentation files in this repository.
