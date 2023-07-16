import streamlit as st
import pickle
import string
import sklearn
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import requests
from streamlit_lottie import st_lottie

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('indonesian') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title='MariNgecek',page_icon='💌')

st.title("MariNgecek🔍")

# --- Load Asset ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_GEUKDGZQL5.json")

# --- Prediction ---
input_sms = st.text_area("Masukkan SMS")

if st.button('Prediksi SMS'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header("SMS Normal (NOT SPAM)")
        lottie_coding = load_lottieurl("https://assets6.lottiefiles.com/private_files/lf30_iwSZjK.json")
    elif result == 1:
        st.header("SMS Penipuan (SPAM)")
        lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_prqvme9e.json")
    else:
        st.header("SMS Promo (SPAM)")
        lottie_coding = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_z4vbrpdr.json")

st_lottie(lottie_coding, height=300,key="email")