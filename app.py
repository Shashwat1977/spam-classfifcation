
import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import sklearn
nltk.download('punkt')
nltk.download('stopwords')


ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Spam Classifier')

input = st.text_area("Enter the message to be classifed")


# 1.Preprocess
# 2.Vectorize

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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


if st.button('Predict'):
    txt = transform_text(input)
    txt = [txt]
    vector = tfidf.transform(txt)
    result  = model.predict(vector)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam/Ham")
