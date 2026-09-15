import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Оценка недвижимости", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"
CSV_FILE_PATH = "real_estate_processed.csv"


@st.cache_data
def load_data():
    try:
        return pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        st.error(f"Не удалось загрузить файл {CSV_FILE_PATH}: {e}")
        return None


df = load_data()

tab_predict, tab_dashboard, tab_help = st.tabs([
    "Оценка недвижимости",
    "Дашборд",
    "Справка"
])

with tab_predict:
    st.header("Оценка рыночной стоимости")
    st.write("Введите параметры объекта для расчета стоимости.")

    with st.form(key="property_form"):
        total_area = st.number_input("Общая площадь, м² (size)", min_value=5.0, max_value=2000.0, value=85.0)
        rooms = st.number_input("Количество комнат (rooms)", min_value=1, max_value=10, value=2)
        halls = st.number_input("Количество гостиных (halls)", min_value=0, max_value=5, value=1)
        tom = st.number_input("Время на рынке (tom)", min_value=0.0, max_value=500.0, value=12.5)
        floor = st.number_input("Этаж", min_value=1, max_value=100, value=3)
        building_age = st.selectbox("Возраст здания", ["0", "1", "2", "3", "4", "5-10", "11-15", "16-20", "21+"])

        submit_button = st.form_submit_button(label="Рассчитать стоимость")

    if submit_button:
        payload = {
            "items": [
                {
                    "tom": float(tom),
                    "size": float(total_area)
                }
            ]
        }

        with st.spinner("Выполняется расчет..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get("predictions", [])

                    if predictions:
                        price = predictions[0]
                        st.metric(label="Прогнозируемая стоимость", value=f"{price:,.2f} TRY")
                    else:
                        st.warning("Сервер вернул пустой ответ.")
                else:
                    st.error(f"Ошибка сервера (код {response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Ошибка соединения. Убедитесь, что сервер main.py запущен.")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")

with tab_dashboard:
    st.header("Анализ данных")

    if df is not None and not df.empty:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Всего строк", len(df))
        kpi2.metric("Колонок", len(df.columns))

        if 'price_try' in df.columns:
            kpi3.metric("Медианная цена", f"{df['price_try'].median():,.0f} TRY")

        st.subheader("Распределение признака")
        selected_column = st.selectbox("Выберите один признак для построения графика:", num_cols)

        if selected_column:
            chart_data = df[selected_column].dropna().head(500)
            st.bar_chart(chart_data.value_counts().sort_index())

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Зависимость цены от площади")
            if 'size' in df.columns and ('price_try' in df.columns or 'price_try_log' in df.columns):
                target_price = 'price_try_log' if 'price_try_log' in df.columns else 'price_try'
                scatter_df = df[['size', target_price]].dropna().head(500)
                st.scatter_chart(scatter_df, x='size', y=target_price)

        with col_chart2:
            st.subheader("Матрица корреляции")
            if len(num_cols) > 1:
                st.dataframe(df[num_cols].corr(), use_container_width=True)

        st.subheader("Просмотр таблицы")
        st.dataframe(df.head(20), use_container_width=True)

    else:
        st.warning(f"Файл {CSV_FILE_PATH} не найден или пуст.")

with tab_help:
    st.header("Справка по использованию")

    st.subheader("Описание сервиса")
    st.write(
        "Приложение предназначено для расчета стоимости недвижимости на основе обученной модели машинного обучения и анализа датасета.")

    st.subheader("Порядок запуска")
    st.write("1. Запустите сервер API с помощью команды: uvicorn main:app --reload")
    st.write("2. Запустите интерфейс Streamlit с помощью команды: streamlit run app.py")

    st.subheader("Необходимые файлы")
    st.write(
        "В папке проекта должны находиться файлы: main.py, app.py, catboost_model.pkl, scaler.pkl, real_estate_processed.csv")

    st.subheader("Описание основных полей")
    st.write("size: общая площадь объекта в квадратных метрах.")
    st.write("tom: время нахождения объекта на рынке.")
    st.write("price_try: стоимость объекта в лирах.")
    st.write("price_try_log: логарифмированная стоимость.")
    st.write("rooms и halls: количество комнат и гостиных.")