# Next Word Prediction using GRU and Streamlit

## Overview

This project is a Deep Learning based Next Word Prediction System that predicts the most probable next word for a given sequence of text. The model is built using TensorFlow, Keras, and GRU (Gated Recurrent Unit) networks and is deployed through an interactive Streamlit web application.

The application accepts a text sequence from the user and predicts the next likely word based on patterns learned during training.

---

## Features

* Next word prediction using Deep Learning
* GRU-based sequence modeling
* Text preprocessing and tokenization
* Interactive Streamlit web interface
* Real-time prediction
* Modern responsive UI

---

## Tech Stack

### Programming Language

* Python = 3.12.0

### Libraries & Frameworks

* TensorFlow = 2.20.0
* Keras = 3.13.2
* NumPy
* Streamlit
* Pickle

### Deep Learning Architecture

* Embedding Layer
* GRU Layer
* Dropout Layer
* Dense Output Layer

---

## Project Structure

```text
Next-Word-Prediction/
│
├── app.py
├── next_word_model.keras
├── tokenizer.pkl
├── config.pkl
├── requirements.txt
├── experiments.ipynb
└── README.md
```

---

## Model Architecture

The model consists of:

1. Embedding Layer
2. GRU Layer (128 Units)
3. Dropout Layer
4. GRU Layer (64 Units)
5. Dense Layer with Softmax Activation

The model learns contextual relationships between words and predicts the next most probable word in a sequence.

---

## Installation

### Clone the Repository

```bash
git clone : https://github.com/Gaurav20053/Next-word-predictor-using-GRU
cd Next-Word-Prediction
```

### Create Virtual Environment

```bash
python -m venv tfenv
```

### Activate Virtual Environment

Windows:

```bash
tfenv\Scripts\activate
```

Linux/Mac:

```bash
source tfenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Usage

1. Enter a sequence of words.
2. Press Enter or click the Predict button.
3. The model predicts the most likely next word.

### Example

Input:

```text
my love was 
```

Output:

```text
published
```

---

## Training Pipeline

* Data Collection
* Text Cleaning
* Tokenization
* Sequence Generation
* Padding
* Model Training
* Evaluation
* Deployment using Streamlit

---

## Future Improvements

* Top-K word predictions
* Transformer-based language models
* Multi-word text generation
* Enhanced UI/UX
* Deployment on Streamlit Cloud

