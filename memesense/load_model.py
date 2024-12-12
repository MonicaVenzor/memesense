import os

import tensorflow as tf
from colorama import Fore, Style
from tensorflow import keras
from memesense.params import *


def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally
    Return None (but do not Raise) if no model is found

    """
    model_path_complete = os.path.join(model_path, 'modelo_multimodal.keras')

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = keras.models.load_model(model_path_complete)

    print("✅ Model loaded from local disk")

    return latest_model
