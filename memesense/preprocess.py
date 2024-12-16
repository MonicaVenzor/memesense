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

def preprocess_images_and_labels(df, image_folder):
    images, labels = [], []
    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        img = preprocess_image(image_path)
        if img is not None:
            images.append(img)
            labels.append(row['overall_sentiment'])
    return np.array(images), labels

'''def preprocess_text(text):
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
        return None '''

def preprocess_text_bert(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenizar el texto
    max_len = 50
    text_encodings = tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors="tf")
    #print(f"Texto procesado: {text_encodings}")
    return text_encodings['input_ids'], text_encodings['attention_mask']
