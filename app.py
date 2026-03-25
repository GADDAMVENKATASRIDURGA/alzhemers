import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import sys
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from mri_validation import validate_mri_scan

# Set page config
st.set_page_config(
    page_title="Alzheimer's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4ECDC4;
    }
    .success-message {
        background: #D4EDDA;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28A745;
    }
    .warning-message {
        background: #FFF3CD;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions from the original code
def _get_class_names_fallback():
    """Get class names for the model"""
    return ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

def load_trained_model(model_path='best_alzheimer_model.h5'):
    """Load a trained Keras model"""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        loaded = tf.keras.models.load_model(model_path, compile=False)
        st.success(f"✅ Model loaded successfully!")
        return loaded
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    if isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        img = Image.open(image).convert('RGB')
    
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, image, class_names=None, target_size=(224, 224)):
    """Predict single image and return results"""
    if class_names is None:
        class_names = _get_class_names_fallback()

    is_valid, validation_msg = validate_mri_scan(image)
    if not is_valid:
        return None, None, None, validation_msg
    
    x = preprocess_image(image, target_size=target_size)
    preds = model.predict(x, verbose=0)
    probs = preds[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else str(idx)
    return label, float(probs[idx]), probs, None

def create_probability_chart(probs, class_names):
    """Create a probability chart using Plotly"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probs * 100,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Dementia Stage",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

def get_severity_color(prediction):
    """Get color based on prediction severity"""
    severity_map = {
        'NonDemented': '#28A745',      # Green
        'VeryMildDemented': '#FFC107', # Yellow
        'MildDemented': '#FD7E14',     # Orange
        'ModerateDemented': '#DC3545'  # Red
    }
    return severity_map.get(prediction, '#6C757D')

def get_risk_level(prediction):
    """Get risk level description"""
    risk_map = {
        'NonDemented': 'Low Risk - No signs of dementia detected',
        'VeryMildDemented': 'Very Low Risk - Minimal cognitive impairment',
        'MildDemented': 'Moderate Risk - Early signs of dementia',
        'ModerateDemented': 'High Risk - Significant cognitive impairment'
    }
    return risk_map.get(prediction, 'Unknown')

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Alzheimer\'s Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6C757D; font-size: 1.1rem;">AI-Powered MRI Analysis for Early Dementia Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    # Model selection
    model_path = st.sidebar.text_input("Model Path", "best_alzheimer_model.h5")
    
    # Load model
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if st.sidebar.button("🔄 Load Model", type="primary"):
        with st.spinner("Loading model..."):
            st.session_state.model = load_trained_model(model_path)
    
    # Check if model is loaded
    if st.session_state.model is None:
        st.warning("⚠️ Please load the model first using the sidebar.")
        st.info("💡 Make sure 'best_alzheimer_model.h5' is in the same directory as this app.")
        return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Single Image Analysis", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">📸 Upload MRI Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a brain MRI image for analysis (photos or non-MRI images will be rejected)"
        )
        
        if uploaded_file is not None:
            # Display columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🖼️ Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Image info
                st.markdown("### 📋 Image Information")
                st.info(f"""
                - **Filename:** {uploaded_file.name}
                - **Size:** {image.size}
                - **Mode:** {image.mode}
                - **Format:** {image.format}
                """)
            
            with col2:
                st.markdown("### 🔍 Analysis Results")
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    prediction, confidence, probs, validation_msg = predict_image(
                        st.session_state.model, 
                        image, 
                        class_names=_get_class_names_fallback()
                    )

                if prediction is None:
                    st.error("Invalid input. Please provide a valid MRI scanned brain image.")
                    st.stop()
                
                # Display results
                severity_color = get_severity_color(prediction)
                risk_level = get_risk_level(prediction)
                
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {severity_color};">
                    <h3 style="color: {severity_color}; margin-bottom: 0.5rem;">
                        {prediction}
                    </h3>
                    <p style="margin: 0; font-weight: 600;">Confidence: {confidence*100:.1f}%</p>
                    <p style="margin: 0.5rem 0 0 0; color: #6C757D; font-size: 0.9rem;">
                        {risk_level}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                st.markdown("### 📊 Probability Distribution")
                fig = create_probability_chart(probs, _get_class_names_fallback())
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if prediction == 'NonDemented':
                    st.success("✅ No signs of dementia detected. Continue regular check-ups.")
                elif prediction == 'VeryMildDemented':
                    st.warning("⚠️ Minimal cognitive impairment detected. Consider follow-up with a specialist.")
                elif prediction == 'MildDemented':
                    st.warning("⚠️ Early signs of dementia detected. Medical consultation recommended.")
                else:
                    st.error("🚨 Significant cognitive impairment detected. Immediate medical attention advised.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Batch Analysis</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose multiple MRI images...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple brain MRI images for batch analysis (non-MRI images will be rejected)"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files uploaded")
            
            if st.button("🚀 Analyze All Images", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Analyzing {file.name}...")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    try:
                        image = Image.open(file)
                        prediction, confidence, probs, validation_msg = predict_image(
                            st.session_state.model, 
                            image, 
                            class_names=_get_class_names_fallback()
                        )
                        if prediction is None:
                            results.append({
                                'Filename': file.name,
                                'Prediction': 'Invalid Image',
                                'Confidence': 'N/A',
                                'Risk Level': validation_msg
                            })
                        else:
                            results.append({
                                'Filename': file.name,
                                'Prediction': prediction,
                                'Confidence': f"{confidence*100:.1f}%",
                                'Risk Level': get_risk_level(prediction)
                            })
                    except Exception as e:
                        results.append({
                            'Filename': file.name,
                            'Prediction': 'Error',
                            'Confidence': 'N/A',
                            'Risk Level': f'Processing error: {str(e)}'
                        })
                
                status_text.text("✅ Analysis complete!")
                
                # Display results as DataFrame
                df = pd.DataFrame(results)
                st.markdown("### 📋 Analysis Results")
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                st.markdown("### 📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                valid_results = [r for r in results if r['Prediction'] not in ['Error', 'Invalid Image']]
                if valid_results:
                    predictions = [r['Prediction'] for r in valid_results]
                    prediction_counts = pd.Series(predictions).value_counts()
                    
                    with col1:
                        st.metric("Total Images", len(valid_results))
                    with col2:
                        st.metric("NonDemented", prediction_counts.get('NonDemented', 0))
                    with col3:
                        st.metric("Mild Cases", prediction_counts.get('MildDemented', 0))
                    with col4:
                        st.metric("Moderate Cases", prediction_counts.get('ModerateDemented', 0))
                
                # Create summary chart
                if valid_results:
                    st.markdown("### 📊 Distribution of Results")
                    fig = px.pie(
                        values=prediction_counts.values,
                        names=prediction_counts.index,
                        title="Prediction Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🧠 What is Alzheimer's Disease?
            
            Alzheimer's disease is a progressive neurological disorder that affects memory, thinking, and behavior. 
            It is the most common cause of dementia, accounting for 60-80% of cases.
            
            ### 🎯 How This App Works
            
            This application uses a deep learning model trained on thousands of MRI brain scans to detect 
            early signs of Alzheimer's disease. The model analyzes brain structure and identifies patterns 
            associated with different stages of cognitive impairment.
            
            ### 📊 Classification Categories
            
            - **NonDemented**: No signs of dementia detected
            - **VeryMildDemented**: Minimal cognitive impairment
            - **MildDemented**: Early signs of dementia
            - **ModerateDemented**: Significant cognitive impairment
            
            ### ⚠️ Important Disclaimer
            
            **This application is for educational and research purposes only and should not be used 
            as a substitute for professional medical diagnosis. Always consult with qualified healthcare 
            professionals for medical concerns.**
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Model Performance
            
            The model has been trained on a comprehensive dataset of MRI scans with the following performance:
            
            - **Accuracy**: ~95%
            - **Precision**: ~95%
            - **Recall**: ~95%
            
            ### 🔧 Technical Details
            
            - **Architecture**: Custom CNN
            - **Input Size**: 224x224 pixels
            - **Classes**: 4 categories
            - **Framework**: TensorFlow/Keras
            """)

if __name__ == "__main__":
    main()
