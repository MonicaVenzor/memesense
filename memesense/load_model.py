import os

from colorama import Fore, Style
from tensorflow import keras
from memesense.params import *


def load_model() -> keras.Model:
    """
    Return a saved model:
    - Tries to load a .keras model first
    - Falls back to a .h5 model if the .keras file cannot be found
    """
    model_keras_path = os.path.join(model_path, 'modelo_multimodal.keras')
    model_h5_path = os.path.join(model_path, 'modelo_multimodal.h5')

    try:
        print(Fore.BLUE + f"\nAttempting to load .keras model from {model_keras_path}..." + Style.RESET_ALL)
        latest_model = keras.models.load_model(model_keras_path)
        print("✅ .keras model loaded successfully!")
        return latest_model
    except:
        print(Fore.YELLOW + f"\nAttempting to load .h5 model from {model_h5_path}..." + Style.RESET_ALL)
        latest_model = keras.models.load_model(model_h5_path)
        print("✅ .h5 model loaded successfully!")
        return latest_model
