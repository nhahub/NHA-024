import streamlit as st
import numpy as np
import tensorflow as tf
import json
import cv2
from PIL import Image
import io

# Configuration
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = (256, 256)

# Page configuration
st.set_page_config(
    page_title="Arabic Sign Language Recognition",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fancy styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 20px;
    }
    h2, h3 {
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 20px 0;
    }
    .result-text {
        font-size: 4rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin: 20px 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        color: #764ba2;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
    }
    .upload-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.5);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_labels():
    """Load the model and labels (cached for performance)"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_map = json.load(f)
        labels = {int(k): v for k, v in labels_map.items()}
        return model, labels
    except Exception as e:
        st.error(f" Failed to load model or labels: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Resize to model input size
    resized_image = cv2.resize(image_np, IMG_SIZE)
    
    # Preprocess for MobileNetV2
    input_arr = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(resized_image, dtype=np.float32)
    )
    
    # Add batch dimension
    input_arr = np.expand_dims(input_arr, axis=0)
    
    return input_arr

def predict_image(model, labels, image):
    """Make prediction on the image"""
    try:
        # Preprocess
        input_arr = preprocess_image(image)
        
        # Predict
        predictions = model.predict(input_arr, verbose=0)
        predicted_class_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        # Get label
        predicted_label = labels.get(predicted_class_idx, "Unknown")
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            (labels.get(int(idx), "Unknown"), float(predictions[0][idx]))
            for idx in top_3_idx
        ]
        
        return predicted_label, confidence, top_3_predictions
    except Exception as e:
        st.error(f" Prediction failed: {str(e)}")
        return None, None, None

def main():
    # Title with emoji
    st.markdown("<h1> Arabic Sign Language Recognition </h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>Upload an image or use your webcam to recognize Arabic sign language letters</p>", unsafe_allow_html=True)
    
    # Load model and labels
    model, labels = load_model_and_labels()
    
    if model is None or labels is None:
        st.error(" Model or labels not loaded. Please check the files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("###  Controls")
        st.markdown("---")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload Image", "📷 Use Webcam"],
            label_visibility="visible"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This application uses a deep learning model to recognize "
            "Arabic sign language letters from images. Simply upload an image "
            "or use your webcam to get started!"
        )
        
        st.markdown("---")
        st.markdown(f"### 📊 Model Info")
        st.success(f"✅ Model loaded successfully")
        st.metric("Total Classes", len(labels))
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Input Image")
        
        image = None
        
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an image of an Arabic sign language letter"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        else:  # Webcam
            camera_image = st.camera_input("Take a picture")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                st.image(image, caption="Captured Image", use_container_width=True)
    
    with col2:
        st.markdown("###  Prediction Results")
        
        if image is not None:
            # Predict button
            if st.button(" Predict Sign Language Letter", type="primary"):
                with st.spinner(" Analyzing image..."):
                    predicted_label, confidence, top_3 = predict_image(model, labels, image)
                
                if predicted_label is not None:
                    # Display main prediction
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h3 style="text-align: center; color: #667eea;">Predicted Letter</h3>
                            <div class="result-text">{predicted_label.upper()}</div>
                            <div class="confidence-text">Confidence: {confidence*100:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Confidence meter
                    st.progress(confidence)
                    
                    # Success message
                    if confidence > 0.8:
                        st.success(" High confidence prediction!")
                    elif confidence > 0.5:
                        st.warning(" Medium confidence prediction")
                    else:
                        st.error(" Low confidence prediction")
                    
                    # Top 3 predictions
                    st.markdown("---")
                    st.markdown("####  Top 3 Predictions")
                    
                    for i, (label, conf) in enumerate(top_3, 1):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{i}. {label.upper()}**")
                        with col_b:
                            st.write(f"{conf*100:.1f}%")
                        st.progress(conf)
        else:
            st.info(" Please upload an image or take a picture using the webcam to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: white; padding: 20px;'>"
        "Made with for Arabic Sign Language Recognition"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
