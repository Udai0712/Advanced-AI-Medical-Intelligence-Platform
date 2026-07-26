from PIL import Image

from app.utils.preprocess import preprocess_pil_image
from app.utils.predict import predict

# Change this path to any image you want to test
image = Image.open("datasets/HAM10000_images_part_1/ISIC_0027419.jpg")

processed = preprocess_pil_image(image)

result = predict(processed)

print(result)