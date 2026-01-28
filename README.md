# Brain Tumor Detection using Deep Learning

A deep learning–based web application for detecting brain tumors from MRI scans.  
This project uses **EfficientNetB0** with transfer learning to classify MRI images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No tumor**

The trained model is deployed using **Streamlit** for an interactive and user-friendly interface.

---

## Demo Video

[![Watch the video](https://img.youtube.com/vi/99SklHe7cYw/hqdefault.jpg)](https://youtu.be/99SklHe7cYw)

---

## Features

- Upload MRI scans (`.jpg`, `.jpeg`, `.png`)
- Automatic tumor classification
- Confidence score with visual progress bar
- Clean and modern Streamlit UI
- Trained on a real MRI dataset with class balancing
- Performance evaluation with accuracy, confusion matrix, ROC curves

---

## Model Overview

- **Architecture:** EfficientNetB0 (Transfer Learning)
- **Input Size:** 128 × 128 RGB
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Training Strategy:**
  - Frozen base model initially
  - Fine-tuning of last 30 layers
  - Class weighting to handle imbalance
  - Early stopping & learning rate reduction

---

# Setup Instructions

## 1) Clone the Repository
```bash

git clone https://github.com/jatin-wig/Brain-Tumor-Detection.git
```

## 2) Install Dependencies
```bash
pip install -r requirements.txt
```

## 3) Run the App
```bash
streamlit run app.py
 ```
or 
```bash
python -m streamlit run app.py 
```
--- 

## Demo

You can access the live demo of the application by visiting the following link:  
[View Demo](https://brain-tumor-detection-jatinwig.streamlit.app/)

# Built by Jatin Wig
### GitHub: https://github.com/jatin-wig

