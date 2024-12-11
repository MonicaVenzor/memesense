# Import
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# Dataset from raw data
df = pd.read_csv('raw_data/memotion_dataset_7k/labels.csv')
print(f"Dataset shape: {df.shape}")

# Cleaning data
## NA values
df_cleaned = df.drop(columns=['text_ocr', 'Unnamed: 0'])
df_cleaned = df_cleaned.dropna(subset=['text_corrected'])
print(f"Duplicates found: {df_cleaned.duplicated().sum()}")

## Text preprocessing
df_cleaned['text_cleaned'] = df_cleaned['text_corrected'].str.lower().str.replace('[^\w\s]', '', regex=True)
df_cleaned = df_cleaned[df_cleaned['text_cleaned'].str.strip() != ""]
df_cleaned.drop(columns=['text_corrected'], inplace=True)

##Combine overall_sentiment labels very_positive > positive & very_negative > negative
df_cleaned['overall_sentiment'] = df_cleaned['overall_sentiment'].replace('very_positive', 'positive')
df_cleaned['overall_sentiment'] = df_cleaned['overall_sentiment'].replace('very_negative', 'negative')

## Working with images
image_folder = 'raw_data/memotion_dataset_7k/images/'
df_cleaned['image_exists'] = df_cleaned['image_name'].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))
df_cleaned = df_cleaned[df_cleaned['image_exists']]

## Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load, resize and normalize an image.
    """
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, target_size)  # Resize images
        img = img / 255.0  # NormNormalize images
    return img

## Preprocess images and collect labels
images = []
labels = []
for idx, row in df_cleaned.iterrows():
    image_path = os.path.join(image_folder, row['image_name'])
    if os.path.exists(image_path):
        img = preprocess_image(image_path)
        if img is not None:
            images.append(img)
            labels.append(row['overall_sentiment'])  # General Sentiment Label

images = np.array(images)
print(f"Processed images: {images.shape}")

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 50
text_encodings = tokenizer(list(df_cleaned['text_cleaned']), truncation=True, padding=True, max_length=max_len, return_tensors="tf")

# Label encoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Convertir etiquetas a números
print(f"Encoder labels: {len(set(y))}")

# Split
## Ensure all arrays have the same number of samples
min_samples = min(len(images), len(text_encodings['input_ids']), len(y))
images = images[:min_samples]
text_encodings['input_ids'] = text_encodings['input_ids'][:min_samples]
text_encodings['attention_mask'] = text_encodings['attention_mask'][:min_samples]
y = y[:min_samples]

## Split data
X_train_images, X_test_images, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)
text_data_train, text_data_test, attention_mask_train, attention_mask_test = train_test_split(
    text_encodings['input_ids'].numpy(),
    text_encodings['attention_mask'].numpy(),
    test_size=0.2,
    random_state=42
)

## Transform to tensors
text_data_train = tf.convert_to_tensor(text_data_train)
text_data_test = tf.convert_to_tensor(text_data_test)
attention_mask_train = tf.convert_to_tensor(attention_mask_train)
attention_mask_test = tf.convert_to_tensor(attention_mask_test)

# Model
## multimodal model
image_input = Input(shape=(224, 224, 3), name="Image_Input")
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))(image_input)
resnet_out = GlobalAveragePooling2D()(resnet_base)

text_input_ids = Input(shape=(max_len,), dtype=tf.int32, name="Text_Input_Ids")
text_attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="Text_Attention_Mask")

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=tf.cast(attention_mask, tf.float32))
        return outputs[1]

bert_layer = BertLayer()
bert_out = bert_layer([text_input_ids, text_attention_mask])

combined = Concatenate()([resnet_out, bert_out])
x = Dense(256, activation='relu')(combined)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(set(y_train)), activation='softmax')(x)

model = Model(inputs=[image_input, text_input_ids, text_attention_mask], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## Train model
history = model.fit(
    [X_train_images, text_data_train, attention_mask_train],
    y_train,
    validation_data=([X_test_images, text_data_test, attention_mask_test], y_test),
    epochs=20,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

## Test
test_loss, test_accuracy = model.evaluate([X_test_images, text_data_test, attention_mask_test], y_test)
print(f"Pérdida en prueba: {test_loss}")
print(f"Precisión en prueba: {test_accuracy}")

## Save model
model.save('modelo_multimodal.h5')

# Prediction on the test set
predictions = model.predict([X_test_images, text_data_test, attention_mask_test])
class_predict = predictions.argmax(axis=1)

## Decode predicted labels
decoded_predictions = label_encoder.inverse_transform(class_predict)
decoded_actuals = label_encoder.inverse_transform(y_test)

## Show some real predictions and labels
for i in range(5):
    print(f"Image {i+1}: Prediction - {decoded_predictions[i]}, Real label - {decoded_actuals[i]}")

# Classification report
print("Classification report:")
print(classification_report(y_test, class_predict, target_names=labels))
