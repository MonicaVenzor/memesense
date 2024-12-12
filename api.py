from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from memesense.load_model import load_model
from memesense.main import predict

import numpy as np

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
    ### Receiving and decoding the image
    contents = await image.read()
    ### Predict label
    model = app.state.model
    label_prediction = predict(image, model)
    return {'label':(label_prediction[0])}
