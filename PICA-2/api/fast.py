from fastapi import FastAPI, Response , File , UploadFile , Request
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
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from datetime import date
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter , _rate_limit_exceeded_handler
from slowapi.util import get_remote_address


###API TITEL
app = FastAPI(title = "PICA2 AMAZING GANs")
limiter = Limiter(key_func = get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded , _rate_limit_exceeded_handler)

@dataclass
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
    app.state.categories = ["house","apple","mountain","squirrel","door","cloud", "butterfly","smiley"]
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
@limiter.limit("5/10seconds")
async def get_discriminated(request : Request , image : UploadFile , category : str = "apple"):
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


#####

def resize_and_colour(img, cmap):
    image = cv2.resize(img, [448,448], interpolation=cv2.INTER_CUBIC)
    fig = plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig('one.png', bbox_inches = 'tight')
    image = plt.imread('one.png')
    return image

def signature(image,cmap,name, x=400,x1=0,y=450,y1=450):
    today = date.today()
    user_input = name
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.scatter(x,y,clip_on=False,color="white",alpha=0.00001)
    plt.scatter(x1,y1,clip_on=False,color="white",alpha=0.00001)
    plt.text(x,y,user_input)
    plt.text(x1,y1,today)
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.savefig('two.png', bbox_inches = 'tight')
    image = plt.imread('two.png')
    return image

def resize_color_signed(img,cmap,name, x=400,x1=0,y=450,y1=450):
    image = cv2.resize(img, [448,448], interpolation=cv2.INTER_CUBIC)
    today = date.today()
    plt.scatter(x,y,clip_on=False,color="white",alpha=0.00001)
    plt.scatter(x1,y1,clip_on=False,color="white",alpha=0.00001)
    plt.text(x,y,name)
    plt.text(x1,y1,today)
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.savefig('one.png', bbox_inches = 'tight')
    image = plt.imread('one.png')
    return image


def blurr_image(img1):
    #takes in image array, gives out image array that is blurred
    flattened_image = img1.reshape(784,1)
    blurred = gaussian_filter(flattened_image,sigma=7)
    output = blurred.reshape(28,28,1)
    return output


def image_numpy_mixer(img1, img2, strength):
    #First image conversion
    np_frame_1 = np.array(img1)
    np_frame_1 = np_frame_1.reshape(28,28,1)
    #Second image conversion
    np_frame_2 = np.array(img2)
    np_frame_2 = np_frame_2.reshape(28,28,1)
    #Mixer of numpy
    new_img = (np_frame_1*(1-strength)+(np_frame_2*strength))
    return new_img

def generate_image(category : str, noise_dim: int = 100,num_examples: int = 1):
    seed = normal([num_examples, noise_dim])
    gans = app.state.checkpoints[category].generator(seed, training=False).numpy()[0, :, :, 0]
    return (cv2.resize(gans, (28, 28), interpolation=cv2.INTER_AREA)*255).astype(np.uint8)

def discriminate_image(category : str , img : np.ndarray) -> np.ndarray:
    img_arr = np.array(img, dtype=np.uint8)
    #nparr = np.fromstring( await image.read() , dtype = np.uint8)
    #nparr = cv2.imdecode(nparr , cv2.IMREAD_COLOR)[1]
    try:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2GRAY)
    except cv2.error:
        pass

    #img = cv2.cvtColor(img_arr , cv2.COLOR_RGBA2GRAY)
    img_resized = cv2.resize(img_arr, (28, 28), interpolation=cv2.INTER_AREA)
    img_resized = (img_resized - 127.5) / 127.5
    prediction = app.state.checkpoints[category].discriminator(img_resized.reshape(1, *img_resized.shape, 1))
    return prediction

@app.post("/super" , response_class = Response)
@limiter.limit("5/minute")
async def get_super(request : Request ,
                    alpha: str ,
                    beta : str,
                    color : str ,
                    name : str, #I added this I hope that's cool
                    noise_dim: int = 100,
                    num_examples: int = 1):
    # Generate and save a random image as Supercookie
    if alpha not in app.state.checkpoints.keys() or beta not in app.state.checkpoints.keys():
        return {"Error": "Invalid Category"}

    initial_image = generate_image(alpha , noise_dim , num_examples)
    add_image_alpha = generate_image(alpha, noise_dim , num_examples)
    add_image_beta = generate_image(beta, noise_dim , num_examples)

    #success, encoded_image = cv2.imencode('.png', gans[0,:,:,0])
    #if success:g
    #    image = encoded_image.tobytes()

     #Parameters
    alpha_fakeness = -7
    beta_fakeness = -7
    strength = 0.1

    # Compare the Supercookie image to both discriminants
    a_pred = discriminate_image(alpha, initial_image)
    b_pred = discriminate_image(beta, initial_image)

    condition = (a_pred < alpha_fakeness and b_pred < beta_fakeness)
    max_iter = 500
    iter = 0

    # Compare to thresholds and update
    while condition == False:
        if a_pred >= alpha_fakeness:
            initial_image = image_numpy_mixer(initial_image, add_image_alpha, strength)
            a_pred = discriminate_image(alpha, initial_image)
            b_pred = discriminate_image(beta, initial_image)
            condition = (a_pred < alpha_fakeness and b_pred < beta_fakeness)
            iter += 1

        if b_pred >= beta_fakeness:
            initial_image = image_numpy_mixer(initial_image, add_image_beta, strength)
            a_pred = discriminate_image(alpha, initial_image)
            b_pred = discriminate_image(beta, initial_image)
            condition = (a_pred < alpha_fakeness and b_pred < beta_fakeness)
            iter += 1

        if condition == True:
            break

    print(type(initial_image))
    print(initial_image.shape)

    #scale and colour: array to array
    #initial_image = resize_and_colour(initial_image, cmap=color)
    #add name and date: array to array
    #initial_image = signature(initial_image,cmap=color,name=name, x=400,x1=0,y=450,y1=450)

    #New scale,color,sign
    initial_image = resize_color_signed(initial_image,cmap=color,name=name, x=400,x1=5,y=440,y1=440)

    #Return and display
    resized_img = (initial_image*127.5+127.5).astype(np.uint16)

    #resized_img = (cv2.resize(initial_image, (28, 28), interpolation=cv2.INTER_AREA)*127.5+127.5).astype(np.uint16)
    im = cv2.imencode('.png', resized_img)[1] #OG

    return Response(content=im.tobytes(), media_type="image/png") #OG













    #initial_image = (initial_image*127.5+127.5).astype(np.uint16)
    #im = cv2.imencode('.png', initial_image)[1]
    #return Response(content=im.tobytes(), media_type="image/png")
    #image.save(bytes_image, format='PNG')
    #return Response(content=bytes_image.getvalue(), media_type="image/png")
    #im = cv2.imencode('.png', resized_img)[1]
    #initial_image = color_under_image(color, initial_image)
    #resized_img = (cv2.resize(initial_image, (28, 28), interpolation=cv2.INTER_AREA)*127.5+127.5).astype(np.uint16)
    #image = Image.fromarray(resized_img)
    #success, encoded_image = cv2.imencode('.png', gans[0,:,:,0])
    #if success:g
    #image = encoded_image.tobytes()
    #bytes_image = io.BytesIO()
