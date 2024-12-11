from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io


import numpy as np

app = FastAPI()

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

@app.post('/upload_image')
async def receive_image(image: UploadFile = File(...), text: str = Form(...)):
    ### Receiving and decoding the image
    #contents = await img.read()

    #nparr = np.fromstring(contents, np.uint8)
    #return {'wait': 64}

     # Process the string
    print(text)
    # ... do something with the string ...

    return {"message": "Data processed successfully"}
