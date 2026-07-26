import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 224

def preprocess_pil_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image).astype("float32")

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image