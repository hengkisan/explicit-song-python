import streamlit as st
import pickle
import numpy as np
from PIL import Image

st.title('Explicit Song Analyser')
st.write('This app uses Random Forest to classify whether song lyrics contain explicit contents')

form = st.form(key='explicit-form')
user_input = form.text_area('Enter song lyrics')

submit = form.form_submit_button('Submit')

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('rf_model_bal.pkl', 'rb') as file:
    clf = pickle.load(file)

if submit:

    user_input = [user_input]

    user_input_to_model = vectorizer.transform(user_input)

    result = clf.predict(user_input_to_model)

    if result == 1:
        st.error('Explicit')
    elif result == 0:
        st.success('Non-Explicit')
