import streamlit as st
import pandas as pd
import pickle
from sklearn.neural_network import MLPRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_model():
    model = MLPRegressor()
    with open('models/model_reg.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

st.title("Веб приложение ML")

# Виджет для загрузки файла
uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:")
        st.dataframe(df_uploaded)

        model = load_model()
        predictions = model.predict(df_uploaded.drop("price", axis=1))
        df_uploaded["Predicted_Price"] = predictions

        st.write("Результаты предсказания:")
        st.dataframe(df_uploaded)

        st.download_button(
            label="Скачать результаты предсказания",
            data=df_uploaded.to_csv(index=False).encode("utf-8"),
            file_name="predicted_results.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке файла: {e}")

else:
    st.subheader("Введите данные для предсказания:")

    input_data = {}
    feature_names = [
       "Year", "Age", "Transmission_Automatic", "Engine_capacity", "Distance"	
    ]

    for feature in feature_names:
        input_data[feature] = st.text_input(f"Введите значение для {feature}")

    if st.button("Сделать предсказание"):
        try:
            input_df = pd.DataFrame([input_data])
            model = load_model()
            y_pred = model.predict(input_df)
            print(y_pred)
            st.success("Предсказанная цена: {:.2f}".format(y_pred[0]))
        except Exception as e:
            st.error(f"Произошла ошибка при предсказании: {e}")
