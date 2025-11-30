#  Natiq - Arabic Sign Language Recognition

<div align="center">

      ███╗   ██╗ █████╗ ████████╗██╗ ██████╗ 
      ████╗  ██║██╔══██╗╚══██╔══╝██║██╔═══██╗
      ██╔██╗ ██║███████║   ██║   ██║██║   ██║
      ██║╚██╗██║██╔══██║   ██║   ██║██║▄▄ ██║
      ██║ ╚████║██║  ██║   ██║   ██║╚██████╔╝
      ╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚══▀▀═╝ 
     



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

</div>

---

## About The Project

**Natiq** (ناطق - meaning "speaker" in Arabic) is an AI-powered application that recognizes Arabic sign language letters from images. Using deep learning and computer vision, this project helps bridge communication gaps for the deaf and hard-of-hearing Arabic-speaking community.

###  Features

-  **28 Arabic Letters Recognition** - Recognizes all Arabic sign language letters
-  **Multiple Input Methods** - Upload images or use your webcam
-  **Deep Learning Powered** - Uses MobileNetV2 for accurate predictions
-  **User-Friendly Interface** - Beautiful Streamlit web interface
-  **Confidence Scores** - Shows prediction confidence and top 3 results
- **Fast & Efficient** - Quick predictions with optimized model

---

##  Project Workflow

![Project Workflow](workflow.png)

---

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (optional, for real-time recognition)

### Installation

1. **Clone or download this project**

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   
   **On Windows (PowerShell):**
   ```powershell
   .\start_gui.ps1
   ```
   
   **Or manually:**
   ```bash
   streamlit run gui_app.py
   ```

4. **Open your browser** - The app will automatically open at `http://localhost:8501`

---

##  How to Use

1. **Launch the application** using the steps above
2. **Choose your input method:**
   -  Upload an image of an Arabic sign language letter
   -  Use your webcam to capture a sign
3. **Click "Predict"** to see the results
4. **View the prediction** with confidence score and top 3 alternatives

---

##  Project Structure

```
Natiq/
├── gui_app.py             # Main Streamlit application
├── Model Training.py      # Model training source code        
├── model.zip              # Trained deep learning model
├── labels.json            # Arabic letter labels mapping
├── requirements.txt       # Python dependencies
├── start_gui.ps1          # Windows launcher script
├── workflow.png           # Project workflow diagram
└── README.md              # This file
```

---

## Recognized Letters

The model can recognize all **28 Arabic sign language letters**:

| Letter | Arabic | Letter | Arabic | Letter | Arabic |
|--------|--------|--------|--------|--------|--------|
| Alif   | أ      | Jiim   | ج      | Saad   | ص      |
| Baa    | ب      | Haa    | ح      | Daad   | ض      |
| Ta     | ت      | Kha    | خ      | Taa    | ط      |
| Tha    | ث      | Daal   | د      | Zaa    | ظ      |
| ... and more!

---

##  Technologies Used

- **Python** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **MobileNetV2** - Pre-trained model architecture
- **Streamlit** - Web interface framework
- **OpenCV** - Image processing
- **NumPy** - Numerical computations

---

## Model Information

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Input Size:** 256x256 pixels
- **Classes:** 28 Arabic letters
- **Framework:** TensorFlow/Keras

---

##  Contributing

 Feel free to:
- Report bugs
- Suggest new features
- Improve the documentation
- Share your feedback

---


## Authors

- Youssef Wael
- Shahd tamer
- Malak Essam 
- Alaa Atef 
- Ashraf Amr




