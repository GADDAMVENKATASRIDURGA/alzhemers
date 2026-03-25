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

### Single Image Analysis

1. Click on the "📸 Single Image Analysis" tab
2. Upload an MRI image (JPG, JPEG, PNG, BMP)
3. View the analysis results:
   - Prediction category with confidence score
   - Probability distribution chart
   - Risk assessment and recommendations

### Batch Analysis

1. Click on the "📊 Batch Analysis" tab
2. Upload multiple MRI images at once
3. Click "Analyze All Images"
4. Review:
   - Detailed results table
   - Summary statistics
   - Distribution pie chart

## 🎯 Model Information

- **Architecture**: Custom Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels
- **Classes**: 4 categories
  - NonDemented
  - VeryMildDemented
  - MildDemented
  - ModerateDemented
- **Performance**: ~95% accuracy on test dataset

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **Visualizations**: Plotly & Matplotlib
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: Pillow (PIL)
- **Data Handling**: NumPy & Pandas

## 📱 App Structure

```
project_guntur/
├── app.py                 # Main Streamlit application
├── run_app.py            # Launcher script with checks
├── requirements_ui.txt   # UI dependencies
├── README_UI.md         # This file
├── disease.py           # Original training script
└── best_alzheimer_model.h5  # Trained model (required)
```

## ⚠️ Important Disclaimer

**This application is for educational and research purposes only and should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical concerns.**

## 🔧 Troubleshooting

### Model Not Found
- Ensure `best_alzheimer_model.h5` is in the same directory as `app.py`
- If you have a different model file, update the path in the sidebar

### Package Installation Issues
- Update pip: `pip install --upgrade pip`
- Install packages individually: `pip install streamlit plotly tensorflow`

### Port Already in Use
- The app uses port 8501 by default
- Change port: `streamlit run app.py --server.port 8502`

### Memory Issues
- Reduce batch size in the original training script
- Close other applications to free up memory

## 🎨 Customization

### Changing Colors/Theme
Edit the CSS in `app.py` under the "Custom CSS" section.

### Adding New Features
The app is modular - you can easily add:
- New prediction models
- Additional image formats
- Export functionality
- User authentication

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the original `disease.py` for model details
3. Ensure all dependencies are properly installed

---

**Made with ❤️ for medical AI research**
