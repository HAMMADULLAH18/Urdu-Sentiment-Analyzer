import pickle
import re
import streamlit as st

# -----------------------------
# 1. Load the saved model & vectorizer
# -----------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Optional: use LabelEncoder if saved
try:
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
        use_le = True
except FileNotFoundError:
    label_mapping = {0: "negative", 1: "positive"}
    use_le = False

# -----------------------------
# 2. Preprocessing Functions
# -----------------------------
def normalize_urdu_text(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

urdu_stopwords = set([
    "میں","نے","کو","کا","کی","کے","سے","پر","اور","ہے","ہیں",
    "یہ","وہ","ایک","بھی","ہی","تو","کہ","تھا","تھی","تھے",
    "ہیں","ہو","ہوگا","ہوگی"
])

def remove_urdu_stopwords(tokens):
    return [word for word in tokens if word not in urdu_stopwords]

def preprocess_text(text):
    clean_text = normalize_urdu_text(text)
    tokens = clean_text.split()
    tokens_nostop = remove_urdu_stopwords(tokens)
    final_text = " ".join(tokens_nostop)
    return final_text

# -----------------------------
# 3. Prediction Function
# -----------------------------
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    X = vectorizer.transform([processed_text])
    pred = model.predict(X)
    if use_le:
        label = le.inverse_transform(pred)[0]
    else:
        label = label_mapping[pred[0]]
    return label

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Urdu Sentiment Analyzer", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎬 Urdu Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter a movie review in Urdu to see if it's Positive or Negative.</p>", unsafe_allow_html=True)

user_input = st.text_area("Write your Urdu review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        if sentiment.lower() == "positive":
            st.success(f"Predicted Sentiment: {sentiment} ✅")
        else:
            st.error(f"Predicted Sentiment: {sentiment} ❌")
    else:
        st.warning("Please enter some text first!")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "Created by <b>HammadUllah_2026</b> | Hosted on Streamlit Cloud 🚀"
    "</div>",
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "Created by <b>HammadUllah_2026</b> | Hosted on Streamlit Cloud 🚀"
    "</div>",
    unsafe_allow_html=True
)

# Add GitHub + LinkedIn buttons
st.markdown(
    """
    <div style='text-align: center;'>
        <a href='https://github.com/HAMMADULLAH18' target='_blank'>
            <img src='https://img.shields.io/badge/GitHub-Profile-black?logo=github' style='margin-right: 10px;'/>
        </a>
        <a href='https://www.linkedin.com/in/hammadullah18/' target='_blank'>
            <img src='https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin'/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)