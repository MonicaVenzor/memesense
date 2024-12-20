import numpy as np
import io

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from memesense.load_model import load_model_meme
from memesense.main import extract_text
from memesense.preprocess import preprocess_image, preprocess_text_bert
from memesense.params import *

app = FastAPI()

app.state.model, model_target = load_model_meme()

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
    print(text_proc)
    text_proc, mask_text = preprocess_text_bert(text_proc)
    if text_proc is None:
        raise HTTPException(status_code=400, detail="Text preprocessing failed")

    # Cargar el modelo y realizar la predicción
    model = app.state.model

    if  model_target == 'bert-base-uncased':
        image_proc = np.expand_dims(image_proc, axis=0)
        #text_proc = np.expand_dims(text_proc, axis=0)
        #mask_text = np.expand_dims(mask_text, axis=0)
        print(image_proc.shape)
        print(text_proc.shape)
        print(mask_text.shape)
        label_prediction = model.predict([image_proc, text_proc, mask_text])
        print(label_prediction)
        return {'label': int(label_prediction[0].argmax())}
