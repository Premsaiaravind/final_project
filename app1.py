from tensorflow.keras.models import load_model
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model  # Add this line
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string

import os
from flask import Flask, request, render_template
import json
import pickle
import nltk
import os


# Load the saved Keras model and tokenizer
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Flask app
app = Flask(__name__)

# Real-time data testing route
@app.route('/')
def index():
    return render_template('index.html')
MAXLEN = 200

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['text']
        sentiment = predict_sentiment(review)
        return render_template('results.html', review=review, sentiment=sentiment)

# real-time data testing function
def predict_sentiment(review):
    review = tokenizer.texts_to_sequences([review])
    review = pad_sequences(review, padding='post', maxlen=MAXLEN)  # Change this line
    sentiment = model.predict(review)[0, 0]
    if sentiment > 0.5:
        return "Positive"
    else:
        return "Negative"


if __name__ == '__main__':
    app.run(debug=True)
