import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer

from memesense.params import *

# Preprocess images

def preprocess_image(image, target_size=(224, 224)):
    try:
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image file: {image}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0

        # Ensure the image is float32 and has the correct shape
        img = img.astype('float32')
        return img
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

def preprocess_text(text):
    try:
        # Ensure text is a list, even if it's a single string
        if isinstance(text, str):
            text = [text]

        # Tokenización
        tokenizer = Tokenizer(num_words=10000)  # Máximo número de palabras únicas
        tokenizer.fit_on_texts(text)

        # Convertir a secuencias
        sequences = tokenizer.texts_to_sequences(text)

        # Padding
        max_len = 50  # Longitud máxima de las secuencias
        text_data = pad_sequences(sequences, maxlen=max_len)

        print(f"Texto procesado: {text_data.shape}")
        return text_data
    except Exception as e:
        print(f"Text preprocessing error: {e}")
        return None
