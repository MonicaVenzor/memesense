import io
import numpy as np

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from memesense.load_model import load_model
from memesense.main import extract_text
from memesense.preprocess import preprocess_image, preprocess_text
from memesense.params import *


app = FastAPI()

app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"status": "ok"}


@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    # Lista de extensiones permitidas
    extensions = ['.jpg', '.png', '.jpeg']
    #file_ext = os.path.splitext(image.filename)[1].lower()
    #print(file_ext)
    # Validar la extensión del archivo
    #if file_ext not in extensions:
    #    raise HTTPException(status_code=400, detail="Invalid file type")
    file_bytes = await image.read()

    # Convert the file to an image using PIL
    image = Image.open(io.BytesIO(file_bytes))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Guardar el archivo localmente
    local_filename = f'imagen{extensions[1]}'
    save_path = image.filename
    print(save_path)

    # Read the file bytes and save the image
    with open(local_filename, "wb") as f:
        f.write(file_bytes)

    # Procesar la imagen
    image_proc = preprocess_image(local_filename)
    if image_proc is None:
        raise HTTPException(status_code=400, detail="Image preprocessing failed")

    # Extraer texto de la imagen
    text_proc = extract_text(local_filename)
    if text_proc is None:
        raise HTTPException(status_code=400, detail="Text extraction failed")

    # Preprocesar el texto
    text_proc = preprocess_text(text_proc)
    if text_proc is None:
        raise HTTPException(status_code=400, detail="Text preprocessing failed")

    # Cargar el modelo y realizar la predicción
    model = app.state.model

    if model_target == 'lstm':
        # Preprocesamiento adicional
        image_proc = np.expand_dims(image_proc, axis=0)
        #text_proc = np.expand_dims(text_proc, axis=0)
        # Realizar predicción
        print(text_proc.shape)
        print(image_proc.shape)
        label_prediction = model.predict([image_proc, text_proc])
    print(label_prediction)
    return {'label': float(label_prediction[0].argmax())}
