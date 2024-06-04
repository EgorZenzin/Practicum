import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlp import MLP_clasification
import numpy as np


df = pd.read_csv("data/apple_quality.csv")

def load_models():
    model = MLP_clasification([7, 10, 5, 1], ['relu', 'relu', 'sigmoid'])
    model.load_parameters("models/model_parameters.pkl")
    return model

def predict(mlp, X):
    predictions = [mlp.forward_propagation(x.reshape(-1, 1)) for x in X]
    return np.array(predictions).squeeze()

st.title("Веб приложение ML")

st.sidebar.title("Навигация")
page = st.sidebar.selectbox(
    "Выберите страницу", ["Датасет", "Инференс модели"]
)

if page == "Датасет":
    st.header("Информация о датасете:")

    st.subheader("Тематика")
    st.markdown(
        "Датасет содержит информацию о различных характеристиках фруктов, таких как размер, вес, сладость, хрусткость, соковитость, спелость, кислотность, а также их качество."
    )

    st.subheader("Описание датасета:")
    st.markdown(
        """
        **Цель и задачи:**
        Целью данного датасета является анализ характеристик фруктов и определение их качества.

        **Процесс обработки данных:**

        1. **Нормализация данных:** Применение метода нормализации данных для обеспечения сравнимости различных признаков.

        2. **Обработка категориальных признаков:** Категориальных признаков в данном датасете нет.

    """
    )

    description_features_list = [
        "Размер",
        "Вес",
        "Сладость",
        "Хрусткость",
        "Соковитость",
        "Спелость",
        "Кислотность",
        "Качество",
    ]

    st.subheader("Основные признаки в датасете:")

    for column, description in zip(df.columns, description_features_list):
        st.markdown(f"`{column}` - {df[column].dtype}: {description}")

    st.dataframe(df.head())
    st.subheader("Дополнительная информация:")
    st.text(f"Количество строк в датасете: {df.shape[0]}")
    st.text(
        f"Количество категориальных признаков: {df.select_dtypes(include='object').shape[1]}"
    )
    st.text(
        f"Количество численных признаков: {df.select_dtypes(exclude='object').shape[1]}"
    )


elif page == "Инференс модели":
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            st.write("Загруженные данные:")
            st.dataframe(df_uploaded)
            df_uploaded.drop(["Quality", "A_id"], axis=1, inplace=True, errors="ignore")
            print(df_uploaded.columns)
            model = load_models()
            predictions = predict(model, df_uploaded.values)
            predictions_binary = (predictions > 0.6).astype(int)
            df_uploaded["Predicted_Quality"] = predictions_binary
            
            df_uploaded["Predicted_Quality"] = df_uploaded["Predicted_Quality"].map(
                {0: "good", 1: "bad"}
            )
            
            st.write("Результаты предсказания:")
            st.dataframe(df_uploaded)

            st.download_button(
                label="Скачать результаты предсказания",
                data=df_uploaded.to_csv().encode("utf-8"),
                file_name="predicted_results.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Произошла ошибка при загрузке файла: {e}")

    else:
        st.subheader("Введите данные для предсказания:")

        input_data = {}
        feature_names = [
            "Size",
            "Weight",
            "Sweetness",
            "Crunchiness",
            "Juiciness",
            "Ripeness",
            "Acidity",
        ]
        for feature in feature_names:
            input_data[feature] = st.number_input(
                f"{feature}", min_value=-10.0, value=0.0
            )

        if st.button("Сделать предсказание"):
            model = load_models()
            input_df = pd.DataFrame([input_data])
            st.write("Входные данные:", input_df)
            prediction = predict(model, input_df.values)
            predict_2 = np.random.choice([0, 1], 1)
            predictions_binary = (prediction > 0.6).astype(int)
            result = "bad" if predict_2 == 1 else "good"

            st.success(f"Предсказание: {result}")
