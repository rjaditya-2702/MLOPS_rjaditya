from fastapi import FastAPI, File, UploadFile, status, HTTPException
from pydantic import BaseModel
from predict import predict_image
import io
from PIL import Image
import numpy as np

app = FastAPI()

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy", "message": "Cat & Dog Classifier API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_cat_dog(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Convert to numpy array
        image_array = np.array(image)

        # Get prediction
        prediction = predict_image(image_array)

        return PredictionResponse(
            class_name=prediction['class'],
            confidence=prediction['confidence'],
            probabilities=prediction['probabilities']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))