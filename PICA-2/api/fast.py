from fastapi import FastAPI, Response
from http import HTTPStatus
import numpy as np
from tensorflow.train import latest_checkpoint , Checkpoint
from tensorflow.random import normal
from tensorflow.keras import optimizers
from models import make_generator_model , make_discriminator_model
import os
import cv2
import io
from dataclasses import dataclass , field
import logging
from PIL import Image

app = FastAPI(title = "PICA2 AMAZING GANs")

@dataclass(slots=True)
class GanConfig:
    
    category : str
    noise_dim : int
    num_examples : int
    
logging.basicConfig(level = logging.INFO,format = "%(message)s")
    
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Goodbye!")

@app.on_event("startup")
async def startup_event():
    checkpoint_dir = "training_checkpoints"
    latest = latest_checkpoint(checkpoint_dir)
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator_optimizer = optimizers.Adam(1e-4)
    discriminator_optimizer = optimizers.Adam(1e-4)
    
    checkpoint = Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    
    checkpoint.restore(latest)
    app.state.checkpoint = checkpoint
    logging.info("Checkpoint has been loaded!")

@app.get("/")
async def root() -> dict:
    """Health check."""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    
@app.get("/test")
def get_test(config : GanConfig):
    return {}

@app.get("/ganify" ,responses = {200: {"content": {"image/png": {}}}}, response_class=Response)
def get_ganification(category : str = "apple" , noise_dim : int = 100 , num_examples : int = 1):
    seed = normal([num_examples, noise_dim])
    gans = app.state.checkpoint.generator(seed , training = False).numpy()[0,:,:,0]
    image = Image.fromarray((cv2.resize(gans, (300,300) , interpolation = cv2.INTER_AREA) *255).astype(np.uint8))
    #success, encoded_image = cv2.imencode('.png', gans[0,:,:,0])
    #if success:
    #    image = encoded_image.tobytes()
    bytes_image = io.BytesIO()
    image.save(bytes_image, format='PNG')
    return Response(content = bytes_image.getvalue() , media_type="image/png")
    return {}
    