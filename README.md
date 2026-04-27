# MemeSense — Multimodal Meme Classifier

Memes are one of the most consumed content formats online, yet one of 
the hardest to moderate automatically. Text alone misses visual context; 
images alone miss language. MemeSense combines both.

This was the final project of the Le Wagon Data Science & AI Bootcamp 
(Oct–Dec 2024), developed by a team of four. I led the project as 
project manager: defined the problem, directed modeling decisions, and 
presented the live demo on YouTube.

## What it does

Classifies memes as positive or negative using a multimodal architecture 
that processes both the image and the embedded text simultaneously. 
Built on BERT for text and ResNet for images, fused into a single 
classification model, deployed via FastAPI and Docker.

## Stack

`Python` `BERT` `ResNet` `FastAPI` `Docker` `Streamlit` `Google Cloud`

## Technical decisions

The core challenge was fusing two different input types — image tensors 
and text embeddings — into a single model. We evaluated multiple 
architectures before selecting BERT + ResNet as the optimal combination. 
A key implementation problem was resolving incompatibilities between 
.keras and .H5 model formats for Docker deployment, which required 
custom serialization handling.

The dataset was built from the Memotion Dataset, with additional 
cleaning and labeling to fit the binary classification objective.

## Try it

[Live demo](https://memesense.streamlit.app/)

## Structure

```
memesense/
├── api/
├── models/
├── notebooks/
├── streamlit_app/
└── README.md
```

## Team

Built by [Mónica Venzor](https://github.com/MonicaVenzor), 
[Alina Colman](https://github.com/AlinaColman), and 
[Gerardo Vargas](https://github.com/GerardoVargas) 
as the final project of Le Wagon Data Science & AI Bootcamp (Oct–Dec 2024).
