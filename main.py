from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
import numpy as np

# Инициализация приложения FastAPI
app = FastAPI(
    title="ML Inference Service",
    description="API для предсказаний с выравниванием порядка признаков",
    version="1.0.0"
)

# 1. Загрузка артефактов модели
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("catboost_model.pkl")
except Exception as e:
    scaler = None
    model = None
    print(f"Предупреждение: Не удалось загрузить модели: {e}")

# 2. Схема входных данных для одного объекта
class FeaturesItem(BaseModel):
    tom: float = Field(..., example=12.5, description="Значение признака tom")
    size: float = Field(..., example=150.0, description="Значение признака size")

# Схема для списка объектов (пакетная обработка)
class PredictRequest(BaseModel):
    items: List[FeaturesItem]

# 3. Эндпоинты API
@app.get("/health")
def healthcheck():
    """Проверка работоспособности сервиса и загрузки моделей."""
    is_ready = scaler is not None and model is not None
    return {
        "status": "ok" if is_ready else "error",
        "models_loaded": is_ready
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if scaler is None or model is None:
        raise HTTPException(
            status_code=500,
            detail="Файлы моделей (scaler.pkl / catboost_model.pkl) не найдены на сервере."
        )

    try:
        raw_data = [item.model_dump() for item in request.items]
        df_input = pd.DataFrame(raw_data)

        required_features = list(scaler.feature_names_in_)
        df_aligned = df_input[required_features]

        X_scaled = scaler.transform(df_aligned)

        # Получаем логарифмированное значение
        log_predictions = model.predict(X_scaled)

        # Переводим из логарифма обратно в лиры через экспоненту
        real_predictions = np.exp(log_predictions)

        return {
            "status": "success",
            "predictions": real_predictions.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при обработке данных: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)