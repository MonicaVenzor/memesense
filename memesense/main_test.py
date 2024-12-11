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

# Paths
data_path = 'raw_data/memotion_dataset_7k/labels.csv'
image_folder = 'raw_data/memotion_dataset_7k/images/'

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

# Preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
    return img

def preprocess_images_and_labels(df, image_folder):
    images, labels = [], []
    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        img = preprocess_image(image_path)
        if img is not None:
            images.append(img)
            labels.append(row['overall_sentiment'])
    return np.array(images), labels

# Prepare data
df_cleaned = load_data(data_path)
images, labels = preprocess_images_and_labels(df_cleaned, image_folder)

# Tokenizer and encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 50
text_encodings = tokenizer(list(df_cleaned['text_cleaned']), truncation=True, padding=True, max_length=max_len, return_tensors="tf")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train-test split
X_train_images, X_test_images, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)
text_data_train, text_data_test, attention_mask_train, attention_mask_test = train_test_split(
    text_encodings['input_ids'].numpy(), text_encodings['attention_mask'].numpy(), test_size=0.2, random_state=42
)

text_data_train = tf.convert_to_tensor(text_data_train)
text_data_test = tf.convert_to_tensor(text_data_test)
attention_mask_train = tf.convert_to_tensor(attention_mask_train)
attention_mask_test = tf.convert_to_tensor(attention_mask_test)

# Define BERT Layer
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=tf.cast(attention_mask, tf.float32))
        return outputs[1]

# Build model
image_input = Input(shape=(224, 224, 3), name="Image_Input")
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))(image_input)
resnet_out = GlobalAveragePooling2D()(resnet_base)

text_input_ids = Input(shape=(max_len,), dtype=tf.int32, name="Text_Input_Ids")
text_attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="Text_Attention_Mask")
bert_layer = BertLayer()
bert_out = bert_layer([text_input_ids, text_attention_mask])

combined = Concatenate()([resnet_out, bert_out])
x = Dense(256, activation='relu')(combined)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(set(y_train)), activation='softmax')(x)

model = Model(inputs=[image_input, text_input_ids, text_attention_mask], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Train model
model.fit(
    [X_train_images, text_data_train, attention_mask_train],
    y_train,
    validation_data=([X_test_images, text_data_test, attention_mask_test], y_test),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Save model
model.save('modelo_multimodal.h5')

# Evaluate model
test_loss, test_accuracy = model.evaluate([X_test_images, text_data_test, attention_mask_test], y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Make predictions
def predict_and_visualize(model, images, text_data, attention_masks, label_encoder, num_samples=5):
    predictions = model.predict([images[:num_samples], text_data[:num_samples], attention_masks[:num_samples]])
    predicted_classes = predictions.argmax(axis=1)
    decoded_classes = label_encoder.inverse_transform(predicted_classes)

    for i in range(num_samples):
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f"Predicted: {decoded_classes[i]}\nActual: {label_encoder.inverse_transform([y_test[i]])[0]}")
        plt.show()

predict_and_visualize(model, X_test_images, text_data_test, attention_mask_test, label_encoder)
