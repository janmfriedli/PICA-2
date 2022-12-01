from fastapi import FastAPI, Response , File , UploadFile
from http import HTTPStatus
import numpy as np
from tensorflow.train import latest_checkpoint , Checkpoint
from tensorflow.random import normal
from tensorflow.keras.optimizers.legacy import Adam
from models import make_generator_model , make_discriminator_model
import os
import cv2
import io
from dataclasses import dataclass , field
import logging
from PIL import Image

###API TITEL
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

def load_checkpoint(category : str):
    checkpoint_dir = os.path.join("central_models",category)
    latest = latest_checkpoint( os.path.join(checkpoint_dir , "training_checkpoints"))
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    optimizer = Adam(1e-4)

    checkpoint = Checkpoint(generator_optimizer=optimizer,
                                 discriminator_optimizer=optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    checkpoint.restore(latest)
    return checkpoint

@app.on_event("startup")
async def startup_event():
    app.state.categories = ["house","apple"]
    app.state.checkpoints = dict()
    for cat in app.state.categories:
        app.state.checkpoints.update({cat : load_checkpoint(cat)})
        logging.info(f"Logged checkpoint for {cat}")
    logging.info("Application Startup COMPLETE!")

###HEALTH CHECK
@app.get("/")
async def root() -> dict:
    """Health check."""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }

###TEST
@app.get("/test")
def get_test(config : GanConfig):
    return {}


###GENERATOR ALL
@app.get("/ganify" ,responses = {200: {"content": {"image/png": {}}}}, response_class=Response)
async def get_ganification(category : str = "apple" , noise_dim : int = 100 , num_examples : int = 1):
    if category not in app.state.checkpoints.keys():
        return {"Error":"Invalid Category"}

    seed = normal([num_examples, noise_dim])
    gans = app.state.checkpoints[category].generator(seed , training = False).numpy()[0,:,:,0]
    image = Image.fromarray((cv2.resize(gans, (28,28) , interpolation = cv2.INTER_AREA) *255).astype(np.uint8))
    #success, encoded_image = cv2.imencode('.png', gans[0,:,:,0])
    #if success:g
    #    image = encoded_image.tobytes()
    bytes_image = io.BytesIO()
    image.save(bytes_image, format='PNG')
    return Response(content = bytes_image.getvalue() , media_type="image/png")

@app.post("/discriminate")
async def get_discriminated(image : UploadFile , category : str = "apple"):
    if category not in app.state.checkpoints.keys():
        return {"Error": "Invalid Category"}

    buf = io.BytesIO(await image.read())
    img = Image.open(buf) #.convert("1")
    img_arr = np.array(img , dtype = np.uint8)

    #nparr = np.fromstring( await image.read() , dtype = np.uint8)
    #nparr = cv2.imdecode(nparr , cv2.IMREAD_COLOR)[1]
    try:
        img_arr = cv2.cvtColor(img_arr , cv2.COLOR_RGBA2GRAY)
    except cv2.error:
        pass

    #img = cv2.cvtColor(img_arr , cv2.COLOR_RGBA2GRAY)
    img_resized = cv2.resize(img_arr , (28,28) , interpolation = cv2.INTER_AREA)
    img_resized = (img_resized - 127.5)/127.5
    prediction = app.state.checkpoints[category].discriminator(img_resized.reshape(1,*img_resized.shape , 1))
    return {"prediction" : float(np.mean(prediction.numpy()))}
