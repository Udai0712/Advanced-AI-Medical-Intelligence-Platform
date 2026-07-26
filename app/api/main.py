from fastapi import FastAPI, UploadFile, File
from PIL import Image

from app.utils.preprocess import preprocess_pil_image
from app.utils.predict import predict

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform"
)

@app.get("/")
def home():
    return {
        "message": "AI Medical Platform Running Successfully"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    image = Image.open(file.file)

    image = preprocess_pil_image(image)

    result = predict(image)

    return result