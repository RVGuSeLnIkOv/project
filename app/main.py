from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse
from model.segmentation_model import load_model
from model.preprocessing import read_nii
import cv2
app = FastAPI()

# Загрузка модели машинного обучения
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Получение данных изображения
    contents = await file.read()

    img_size = 224
    data = read_nii(contents, img_size)
    data = cv2.resize(data, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)

    prediction_result = model.predict(data.reshape(1, img_size, img_size, 1))

    # Вернуть результаты
    return {"result": prediction_result}

@app.get("/")
async def main():
    return FileResponse("app/index.html")
