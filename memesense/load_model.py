import os

from colorama import Fore, Style
from tensorflow import keras
from memesense.params import *

import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from keras.saving import register_keras_serializable
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model

@register_keras_serializable()
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_path='bert-base-uncased', **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert_path = bert_path
        self.bert = None  # Se inicializará en build()

    def build(self, input_shape):
        self.bert = TFBertModel.from_pretrained(self.bert_path)
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=tf.cast(attention_mask, tf.float32))
        return outputs[1]

    def get_config(self):
        config = super().get_config()
        config.update({
            'bert_path': self.bert_path
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_model_meme() -> keras.Model:
    """
    Return a saved model:
    - Tries to load a .keras model first
    - Falls back to a .h5 model if the .keras file cannot be found
    """
    model_keras_path = os.path.join(model_path, 'modelo_multimodal.keras')
    model_h5_path = os.path.join(model_path, 'modelo_multimodal.h5')

    try:
        print(Fore.BLUE + f"\nAttempting to load .keras model from {model_keras_path}..." + Style.RESET_ALL)
        with custom_object_scope({'BertLayer': BertLayer}):
            latest_model = load_model(model_keras_path)
        print("✅ .keras model loaded successfully!")
        return latest_model, "bert-base-uncased"
    except:
        print(Fore.YELLOW + f"\nAttempting to load .h5 model from {model_h5_path}..." + Style.RESET_ALL)
        latest_model = keras.models.load_model(model_h5_path)
        print("✅ .h5 model loaded successfully!")
        return latest_model, "lstm"
