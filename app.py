
import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'stopwords', 'wordnet'], quiet=True)

# Load model
model = joblib.load('Mugimba_model.joblib')
tfidf = joblib.load('Mugimba_tfidf.joblib')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

st.title("🍲 The Roots Restaurant - AI Dish Classifier")
st.write("Enter any dish name or description → get instant category prediction")

user_input = st.text_area("Dish description (example: 'Local free range chicken steamed in peanut sauce with matooke')", height=100)

if st.button("Predict Category"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]

        st.success(f"**Predicted Category: {pred.replace('_', ' ')}**")

        # Business suggestions based on category
        suggestions = {
            'Chicken': 'Pair with matooke or kachumbari. Popular for quick lunches.',
            'Goat': 'Signature luwombo – great for weekend specials and events.',
            'Fish': 'Fresh Lake Victoria tilapia – highlight as healthy/Friday special.',
            'Beef': 'Oxtail or gnut luwombo – premium corporate catering item.',
            'Vegetable_Accompaniment': 'Promote as healthy side or vegan option.'
        }
        st.info(f"💡 Business tip: {suggestions.get(pred, 'Excellent traditional Ugandan choice!')}")
    else:
        st.warning("Please enter a description")
