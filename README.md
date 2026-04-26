# 🧠 MedVision – Medical Image Enhancement & Analysis System

A Streamlit-based application that enhances and analyzes medical images using core **Digital Image Processing (DIP)** techniques.

This project demonstrates how preprocessing improves image quality and helps in better visualization of medical data.

---

## 🚀 Features

* 📷 Upload medical images (X-ray, MRI, CT, etc.)
* ⚙️ Auto-optimization of processing parameters
* 🧹 Noise reduction (Gaussian + Median filtering)
* 🌗 Contrast enhancement using CLAHE
* 🔍 Edge detection (Canny)
* 🧩 Segmentation techniques:

  * Binary Thresholding
  * Otsu Thresholding
  * Adaptive Thresholding
* 🧱 Morphological operations (Dilate, Erode, Open, Close)
* 📊 Image quality metrics:

  * Contrast (Standard Deviation)
  * Sharpness (Laplacian Variance)
  * Entropy
  * Edge Density
* 📈 Histogram comparison (Original vs Enhanced)
* 📦 Export results (PNG + ZIP)
* 🧠 Optional AI-based analysis (Claude Vision API)

---

## 🧠 Processing Pipeline

1. **Grayscale Conversion**
   Converts image into a single intensity channel

2. **Noise Reduction**
   Removes noise using Gaussian and Median filtering

3. **Contrast Enhancement (CLAHE)**
   Improves local contrast without amplifying noise

4. **Sharpening**
   Enhances fine structures

5. **Edge Detection (Canny)**
   Highlights boundaries of structures

6. **Segmentation**
   Separates regions based on intensity

7. **Morphological Operations**
   Refines segmentation results

---

## 🏥 Applications

* Medical image preprocessing before AI models
* Enhancing low-quality scans
* Educational tool for learning DIP concepts
* Visualization of anatomical structures

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**.
> It does **NOT perform medical diagnosis** and should not be used for clinical decision-making.

---

## 🛠️ Tech Stack

* Python
* OpenCV
* Streamlit
* NumPy
* Pillow
* Plotly
* Requests

---

## 📦 Installation

```bash
pip install streamlit opencv-python pillow numpy plotly requests
```

---

## ▶️ Run the Application

```bash
streamlit run medical_image_enhancer.py
```

---

## 📸 Usage

1. Upload a medical image
2. Adjust parameters or use auto-optimization
3. View enhanced outputs:

   * Enhanced Image
   * Edge Detection
   * Segmentation
4. Analyze histogram and metrics
5. Download results

---

## 🧠 AI Analysis (Optional)

To enable AI insights:

1. Get API key from https://console.anthropic.com
2. Enter key in the sidebar
3. Click **Run AI Analysis**

---

## 📁 Output

* Individual processed images (PNG)
* Full pipeline results (ZIP)

---

## 🔮 Future Improvements

* Deep learning-based segmentation (U-Net)
* DICOM format support
* Region of Interest (ROI) detection
* Clinical dataset integration

---

## ⭐ If you found this useful, consider giving a star!
