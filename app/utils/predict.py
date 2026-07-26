import numpy as np
from tensorflow.keras.models import load_model

model = load_model("app/models/skin_model.keras")

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus",
    "Vascular Lesion"
]

def predict(image):

    predictions = model.predict(image, verbose=0)

    index = np.argmax(predictions)

    return {
        "prediction": CLASS_NAMES[index],
        "confidence": float(predictions[0][index]),
        "probabilities": {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        },
        "model": model
    }