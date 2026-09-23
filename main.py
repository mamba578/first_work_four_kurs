from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
import numpy as np

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="API для предсказания стоимости недвижимости",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("catboost_model.pkl")
    print("Успех: Модель и Scaler загружены!")
except Exception as e:
    scaler = None
    model = None
    print(f"Предупреждение: Не удалось загрузить модели: {e}")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/")
def root():
    return {"status": "online", "message": "API работает! Перейдите на /docs"}


# Расширенная схема со всеми признаками из датасета
class FeaturesItem(BaseModel):
    size: float = Field(..., example=120.0, description="Площадь объекта в кв.м")
    room_count_num: float = Field(..., example=3.0, description="Количество комнат")
    tom: Optional[float] = Field(30.0, example=15.0, description="Дней на рынке")
    building_age: Optional[str] = Field("0", example="0", description="Возраст здания")
    total_floor_count: Optional[str] = Field("5", example="5", description="Всего этажей")
    floor_no: Optional[str] = Field("3", example="3", description="Этаж объекта")
    city: Optional[str] = Field("İstanbul", example="İstanbul", description="Город")
    sub_type: Optional[str] = Field(None, example="Rezidans", description="Тип недвижимости")
    heating_type: Optional[str] = Field(None, example="Kombi (Doğalgaz)", description="Тип отопления")


class PredictRequest(BaseModel):
    items: List[FeaturesItem]


@app.get("/health")
def healthcheck():
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
            detail="Файлы моделей не найдены на сервере."
        )

    try:
        raw_data = [item.model_dump() for item in request.items]
        df_raw = pd.DataFrame(raw_data)

        # 1. Определяем полный список колонок модели и скейлера
        if hasattr(model, "feature_names_") and model.feature_names_:
            cb_features = list(model.feature_names_)
        elif hasattr(model, "get_feature_names"):
            cb_features = list(model.get_feature_names())
        else:
            cb_features = list(getattr(scaler, "feature_names_in_", []))

        scaler_features = list(getattr(scaler, "feature_names_in_", cb_features))
        all_cols = list(dict.fromkeys(scaler_features + cb_features))

        # 2. Создаем нулевую матрицу под все колонки
        df_full = pd.DataFrame(0.0, index=df_raw.index, columns=all_cols)

        # Заполняем числовые признаки
        if 'size' in df_full.columns and 'size' in df_raw.columns:
            df_full['size'] = df_raw['size'].fillna(0.0).astype(float)
        if 'room_count_num' in df_full.columns and 'room_count_num' in df_raw.columns:
            df_full['room_count_num'] = df_raw['room_count_num'].fillna(0.0).astype(float)
        if 'tom' in df_full.columns and 'tom' in df_raw.columns:
            df_full['tom'] = df_raw['tom'].fillna(30.0).astype(float)

        # Заполняем категориальные и OHE колонки
        for idx, row in df_raw.iterrows():
            # Обработка категориальных колонок (если CatBoost обучен напрямую на них)
            for cat_col in ['building_age', 'total_floor_count', 'floor_no', 'city']:
                if cat_col in df_full.columns and pd.notna(row.get(cat_col)):
                    df_full.loc[idx, cat_col] = str(row[cat_col])

            # Обработка One-Hot Encoded колонок
            if pd.notna(row.get('city')) and f"city_{row['city']}" in df_full.columns:
                df_full.loc[idx, f"city_{row['city']}"] = 1.0
            if pd.notna(row.get('sub_type')) and f"sub_type_{row['sub_type']}" in df_full.columns:
                df_full.loc[idx, f"sub_type_{row['sub_type']}"] = 1.0
            if pd.notna(row.get('heating_type')) and f"heating_type_{row['heating_type']}" in df_full.columns:
                df_full.loc[idx, f"heating_type_{row['heating_type']}"] = 1.0

        # 3. Масштабирование числовых признаков
        if scaler_features:
            scaled_values = scaler.transform(df_full[scaler_features])
            df_full[scaler_features] = scaled_values

        # 4. Инференс
        df_final = df_full[cb_features]
        X_array = df_final.to_numpy()
        log_predictions = model.predict(X_array)

        real_predictions = np.expm1(log_predictions)

        return {
            "status": "success",
            "predictions": np.round(real_predictions, 2).tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при обработке данных: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)