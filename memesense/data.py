import os
import pandas as pd
from memesense.params import *

# Paths


# Load dataset
def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['text_ocr', 'Unnamed: 0'], errors='ignore')
    df = df.dropna(subset=['text_corrected'])
    df['text_cleaned'] = df['text_corrected'].str.lower().str.replace('[^\w\s]', '', regex=True)
    df = df[df['text_cleaned'].str.strip() != ""]
    df['overall_sentiment'] = df['overall_sentiment'].replace({'very_positive': 'positive', 'very_negative': 'negative', 'neutral': 'negative'})
    df['image_exists'] = df['image_name'].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))
    df = df[df['image_exists']]
    return df
