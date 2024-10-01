import pickle
import streamlit as st

tfidf=pickle.load(open('vectorizer.pkl','rb'))
vc=pickle.load(open('model.pkl','rb'))
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
def text_simplifier(text):
    #convert the message to lowercase
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[i for i in text if i.isalnum()]
    text=[i for i in text if (i not in stopwords.words('english') and i not in string.punctuation)]
    stemmer=PorterStemmer()
    text=[stemmer.stem(i) for i in text]
    return " ".join(text)

st.title('Email/SMS Spam Classifier')
input_sms=st.text_input("Enter the message")

if st.button('Predict'):
    transformed_sms = text_simplifier(input_sms)
    sms_vector = tfidf.transform([transformed_sms])
    result = vc.predict(sms_vector)[0]

    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
