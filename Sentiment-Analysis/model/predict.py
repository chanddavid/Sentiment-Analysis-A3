# predict.py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'sentiment_model.keras')

model = load_model(model_path)

# model = load_model('sentiment_model.keras')

def predict_sentiment(text):
    prob = float(model.predict(tf.constant([text]))[0][0])

    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if sentiment == "Positive" else 1 - prob

    return sentiment, confidence