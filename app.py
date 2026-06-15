import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Style
# =========================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Title */
h1 {
    color: white !important;
    text-align: center;
    font-size: 3rem !important;
    font-weight: bold;
}

/* Labels */
label, p {
    color: white !important;
}

/* Input Box */
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.08);
    color: white;
    border-radius: 12px;
    border: 1px solid #334155;
    padding: 10px;
}

/* Button */
.stButton > button {
    background: linear-gradient(to right, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: bold;
    font-size: 16px;
}

.stButton > button:hover {
    background: linear-gradient(to right, #1d4ed8, #1e40af);
    color: white;
}

/* Success Box */
.stSuccess {
    border-radius: 12px;
}

/* Remove Streamlit Header */
header {
    visibility: hidden;
}

/* Hide Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD FILES
# =========================

# Load model (GRU)
model = load_model('next_word_model.keras', compile=False)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load config (IMPORTANT)
with open('config.pkl', 'rb') as f:
    config = pickle.load(f)

max_sequence_len = config["max_sequence_len"]

# =========================
# PREDICTION FUNCTION
# =========================

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Trim if too long
    token_list = token_list[-(max_sequence_len-1):]

    # Pad
    token_list = pad_sequences([token_list],
                               maxlen=max_sequence_len-1,
                               padding='pre')

    # Predict
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)

    # Convert index → word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "Not found"

# =========================
# STREAMLIT UI
# =========================

st.title("Next Word Prediction (GRU Model)")

with st.form("prediction_form"):
    text = st.text_input("Enter a sequence of words:")
    submitted = st.form_submit_button("Predict Next Word")

if submitted and text.strip():
    predicted_word = predict_next_word(
        model,
        tokenizer,
        text,
        max_sequence_len
    )

    st.success(f"Next word: {predicted_word}")