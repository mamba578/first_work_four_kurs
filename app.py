import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Оценка недвижимости", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"
CSV_FILE_PATH = "real_estate_processed.csv"


@st.cache_data
def load_data():
    try:
        return pd.read_csv(CSV_FILE_PATH, low_memory=False)
    except Exception as e:
        st.error(f"Не удалось загрузить файл {CSV_FILE_PATH}: {e}")
        return None


df = load_data()

tab_predict, tab_dashboard, tab_help = st.tabs([
    "Оценка недвижимости",
    "Динамический Дашборд",
    "Справка"
])

# ---------------------------------------------------------
# ВКЛАДКА 1: ОЦЕНКА НЕБИЖИМОСТИ
# ---------------------------------------------------------
with tab_predict:
    st.header("Оценка рыночной стоимости")
    st.write("Заполните параметры объекта для выполнения расчёта:")

    BUILDING_AGE_OPTIONS = ['0', '1', '2', '3', '4', '5', '6-10 arası', '11-15 arası', '16-20 arası', '21-25 arası',
                            '26-30 arası', '31-35 arası', '36-40 arası', '40 ve üzeri', 'Unknown']
    FLOOR_COUNT_OPTIONS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10-20 arası', '20 ve üzeri', 'Unknown']
    FLOOR_NO_OPTIONS = ['Giriş Katı', 'Yüksek Giriş', 'Bahçe katı', 'Zemin Kat', '1', '2', '3', '4', '5', '6', '7', '8',
                        '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20 ve üzeri', 'En Üst Kat',
                        'Çatı Katı', 'Teras Kat', 'Kot 1', 'Kot 2', 'Kot 3', 'Kot 4', 'Asma Kat', 'Bodrum Kat',
                        'Müstakil', 'Komple', 'Unknown']
    SUB_TYPE_OPTIONS = ['Другое / Не указано', 'Komple Bina', 'Kooperatif', 'Köşk / Konak / Yalı', 'Loft',
                        'Müstakil Ev', 'Prefabrik Ev', 'Rezidans', 'Villa', 'Yalı Dairesi', 'Yazlık', 'Çiftlik Evi']
    HEATING_OPTIONS = ['Другое / Не указано', 'Güneş Enerjisi', 'Jeotermal', 'Kalorifer (Akaryakıt)',
                       'Kalorifer (Doğalgaz)', 'Kalorifer (Kömür)', 'Kat Kaloriferi', 'Klima', 'Kombi (Doğalgaz)',
                       'Kombi (Elektrikli)', 'Merkezi Sistem', 'Merkezi Sistem (Isı Payı Ölçer)', 'Soba (Doğalgaz)',
                       'Soba (Kömür)', 'Yerden Isıtma', 'Yok']
    CITIES_OPTIONS = ['İstanbul', 'Ankara', 'İzmir', 'Aydın', 'Antalya', 'Balıkesir', 'Mersin', 'Muğla', 'Adana',
                      'Bursa', 'Другой город']

    with st.form(key="property_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Характеристики")
            total_area = st.number_input("Площадь, м² (size)", min_value=5.0, max_value=2000.0, value=120.0)
            rooms = st.number_input("Комнат (room_count_num)", min_value=1.0, max_value=20.0, value=3.0, step=1.0)
            tom = st.number_input("Дней на рынке (tom)", min_value=0.0, max_value=500.0, value=30.0)

        with col2:
            st.subheader("Параметры здания")
            building_age = st.selectbox("Возраст (building_age)", BUILDING_AGE_OPTIONS, index=0)
            total_floor_count = st.selectbox("Этажей в доме (total_floor_count)", FLOOR_COUNT_OPTIONS, index=4)
            floor_no = st.selectbox("Этаж объекта (floor_no)", FLOOR_NO_OPTIONS, index=5)

        with col3:
            st.subheader("Локация и удобства")
            city = st.selectbox("Город (city)", CITIES_OPTIONS, index=0)
            sub_type = st.selectbox("Тип (sub_type)", SUB_TYPE_OPTIONS, index=0)
            heating_type = st.selectbox("Отопление (heating_type)", HEATING_OPTIONS, index=8)

        submit_button = st.form_submit_button(label="Рассчитать стоимость")

    if submit_button:
        payload = {
            "items": [
                {
                    "size": float(total_area),
                    "room_count_num": float(rooms),
                    "tom": float(tom),
                    "building_age": str(building_age),
                    "total_floor_count": str(total_floor_count),
                    "floor_no": str(floor_no),
                    "city": str(city) if city != "Другой город" else "Unknown",
                    "sub_type": str(sub_type) if sub_type != "Другое / Не указано" else None,
                    "heating_type": str(heating_type) if heating_type != "Другое / Не указано" else None
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
                        st.success("Расчёт завершён успешно!")
                        st.metric(label="Прогнозируемая стоимость", value=f"{price:,.2f} TRY")
                    else:
                        st.warning("Сервер вернул пустой ответ.")
                else:
                    st.error(f"Ошибка сервера (код {response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Ошибка соединения. Убедитесь, что сервер main.py запущен.")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")

# ---------------------------------------------------------
# ВКЛАДКА 2: ДИНАМИЧЕСКИЙ ДАШБОРД
# ---------------------------------------------------------
with tab_dashboard:
    st.header("Динамический интерактивный анализ")

    if df is not None and not df.empty:
        # --- БЛОК ИНТЕРАКТИВНЫХ ФИЛЬТРОВ ---
        with st.expander("Фильтры данных ", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)

            available_cities = sorted(df['city'].dropna().unique().tolist()) if 'city' in df.columns else []
            selected_cities = f_col1.multiselect("Выберите город(а):", options=available_cities, default=[])

            if 'price_try' in df.columns:
                min_p, max_p = float(df['price_try'].min()), float(df['price_try'].max())
                price_range = f_col2.slider("Диапазон цены (TRY):", min_value=min_p, max_value=max_p, value=(min_p, max_p))
            else:
                price_range = None

            if 'room_count_num' in df.columns:
                min_r, max_r = float(df['room_count_num'].min()), float(df['room_count_num'].max())
                room_range = f_col3.slider("Количество комнат:", min_value=min_r, max_value=max_r, value=(min_r, max_r))
            else:
                room_range = None

        # --- ПРИМЕНЕНИЕ ФИЛЬТРОВ ---
        filtered_df = df.copy()

        if selected_cities:
            filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

        if price_range and 'price_try' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['price_try'] >= price_range[0]) &
                (filtered_df['price_try'] <= price_range[1])
                ]

        if room_range and 'room_count_num' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['room_count_num'] >= room_range[0]) &
                (filtered_df['room_count_num'] <= room_range[1])
                ]

        st.markdown("---")

        # --- МЕТРИКИ ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Найдено объектов", f"{len(filtered_df):,}")

        if 'price_try' in filtered_df.columns and not filtered_df.empty:
            kpi2.metric("Медианная цена", f"{filtered_df['price_try'].median():,.0f} TRY")
            kpi3.metric("Средняя цена", f"{filtered_df['price_try'].mean():,.0f} TRY")
        else:
            kpi2.metric("Медианная цена", "N/A")
            kpi3.metric("Средняя цена", "N/A")

        if 'size' in filtered_df.columns and not filtered_df.empty:
            kpi4.metric("Средний индекс площади", f"{filtered_df['size'].mean():.2f}")
        else:
            kpi4.metric("Средняя площадь", "N/A")

        st.markdown("---")

        if filtered_df.empty:
            st.warning("По выбранным фильтрам не найдено ни одной записи.")
        else:
            # --- ОБНОВЛЕННЫЕ ИСПРАВЛЕННЫЕ ГРАФИКИ ---
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.subheader("Распределение цен объектов")
                if 'price_try' in filtered_df.columns:
                    # Использование Plotly для красивой и понятной гистограммы цен
                    fig_price = px.histogram(
                        filtered_df,
                        x="price_try",
                        nbins=30,
                        title="",
                        labels={"price_try": "Цена (TRY)", "count": "Количество объектов"},
                        color_discrete_sequence=['#4B9CD3']
                    )
                    fig_price.update_layout(
                        xaxis_title="Цена (TRY)",
                        yaxis_title="Количество объектов",
                        margin=dict(l=20, r=20, t=20, b=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig_price, use_container_width=True)

            with col_chart2:
                st.subheader("Зависимость цены от площади")
                if 'size' in filtered_df.columns and 'price_try' in filtered_df.columns:
                    # Фильтруем экстремальные выбросы для наглядности (до 99 перцентиля)
                    q_high_price = filtered_df['price_try'].quantile(0.99)
                    q_high_size = filtered_df['size'].quantile(0.99)

                    scatter_df = filtered_df[
                        (filtered_df['price_try'] <= q_high_price) &
                        (filtered_df['size'] <= q_high_size)
                        ].sample(min(1000, len(filtered_df)), random_state=42)

                    fig_scatter = px.scatter(
                        scatter_df,
                        x="size",
                        y="price_try",
                        labels={"size": "Площадь (нормированная)", "price_try": "Цена (TRY)"},
                        opacity=0.6,
                        color_discrete_sequence=['#5CACE2']
                    )
                    fig_scatter.update_layout(
                        xaxis_title="Индекс площади (size)",
                        yaxis_title="Цена объекта (TRY)",
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            col_chart3, col_chart4 = st.columns(2)

            with col_chart3:
                st.subheader("Топ городов по количеству объявлений")
                if 'city' in filtered_df.columns:
                    top_cities = filtered_df['city'].value_counts().head(10)
                    st.bar_chart(top_cities)

            with col_chart4:
                st.subheader("Средняя цена по комнатам")
                if 'room_count_num' in filtered_df.columns and 'price_try' in filtered_df.columns:
                    room_price = filtered_df.groupby('room_count_num')['price_try'].mean().head(10)
                    st.bar_chart(room_price)

            # --- ТАБЛИЦА ---
            st.subheader("Отфильтрованные данные")
            st.dataframe(filtered_df.head(100), use_container_width=True)

# ---------------------------------------------------------
# ВКЛАДКА 3: СПРАВКА
# ---------------------------------------------------------
with tab_help:
    st.header("Справка по работе с приложением")

    st.subheader("О сервисе")
    st.write(
        "Данный сервис состоит из REST API на FastAPI и пользовательского веб-интерфейса на Streamlit. "
        "Он предназначен для оценки рыночной стоимости недвижимости на основе обученной модели CatBoost "
        "и интерактивного анализа исторических данных датасета."
    )

    st.subheader("Назначение основных файлов проекта")
    st.write("1. main.py — серверной бэкенд на FastAPI, принимающий параметры объектов и возвращающий прогноз цен.")
    st.write("2. app.py — пользовательский веб-интерфейс на Streamlit для ввода данных и визуализации аналитики.")
    st.write("3. catboost_model.pkl — обученная модель машинного обучения CatBoost.")
    st.write("4. scaler.pkl — объект для масштабирования числовых признаков перед подачей в модель.")
    st.write("5. model_columns.pkl — список и порядок колонок, необходимый для корректного преобразования данных.")
    st.write("6. real_estate_processed.csv — обработанный датасет для построения графиков и дашбордов.")
    st.write("7. requirements.txt — список необходимых библиотеки и зависимостей Python.")

    st.subheader("Порядок запуска приложения")
    st.write("Шаг 1. Запустите бэкенд-сервер FastAPI:")
    st.code("uvicorn main:app --reload", language="bash")
    st.write("Шаг 2. Запустите интерфейс Streamlit в отдельном терминале:")
    st.code("streamlit run app.py", language="bash")

    st.subheader("Как пользоваться оценкой стоимости")
    st.write("1. Перейдите на вкладку 'Оценка недвижимости'.")
    st.write("2. Заполните параметры объекта: площадь, количество комнат, возраст здания, этаж, город, тип недвижимости и отопления.")
    st.write("3. Нажмите кнопку 'Рассчитать стоимость'. Ответ с прогнозируемой ценой в TRY отобразится на экране.")

    st.subheader("Как пользоваться дашбордом")
    st.write("1. Перейдите на вкладку 'Динамический Дашборд'.")
    st.write("2. Раскройте блок 'Фильтры данных', чтобы отфильтровать объекты по городу, диапазону цен или количеству комнат.")
    st.write("3. Все графики и ключевые показатели пересчитаются автоматически в реальном времени.")