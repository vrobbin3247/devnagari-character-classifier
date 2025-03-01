[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://devanagari-character-classifier.streamlit.app/)  
![Python](https://img.shields.io/badge/Python-3.12.2-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.42.1-red)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)  
![GitHub Stars](https://img.shields.io/github/stars/vrobbin3247/devnagari-character-classifier?style=social)  
![Last Commit](https://img.shields.io/github/last-commit/vrobbin3247/devnagari-character-classifier)  

📝 Devanagari Character Recognition (Streamlit App)

📌 Overview

🔤 Devanagari Character Recognition App built with TensorFlow & Streamlit to classify handwritten characters from the Devanagari script.

📍 Try it here → Devanagari Character Classifier

📂 Dataset

🔗 Source: UCI Devanagari Handwritten Character Dataset
🖼 Total Images: 92,000
🔢 Classes: 46 (including vowels, consonants & numerals)
📏 Resolution: 32x32 pixels

🏗️ Model Architecture

A CNN-based model with:
	•	Conv2D layers + ReLU activation
	•	Batch Normalization & MaxPooling
	•	Softmax Output Layer (for multi-class classification)

📥 Installation

git clone https://github.com/vrobbin3247/devnagari-character-classifier.git
cd devnagari-character-classifier
pip install -r requirements.txt

▶️ Run the App

streamlit run app.py

Upload an image, and the model will predict the character! 🎯

📊 Model Performance

🎯 Accuracy ~91%
📉 Handles most characters well but has minor confusion with visually similar ones
