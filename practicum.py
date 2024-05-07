import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

data_regression = pd.read_csv('cars_data52.csv')

data_classification = pd.read_csv("apple2.csv")





def load_models():
    model_classification = pickle.load(open('ridge.pkl', 'rb'))
    model_regression = pickle.load(open('log_reg.pkl', 'rb'))
    return model_classification, model_regression

def machine_learning():
    page=st.sidebar.selectbox("Выберите страницу",['Датасеты','Предикт'])
    if (page=="Датасеты"):
        "Информация о датасетах:"
        ""
        "apple_quality.csv:"
        "Size: Размеры яблока."
        "Weight: Масса яблока."
        "Sweetness: Вкус яблока."
        "Crunchiness: Текстура яблока."
        "Juiciness: Влажность яблока."
        "Ripeness: Зрелость яблок."
        "Acidity: Терпкость яблок."
        "Quality: Общая оценка яблок."
        "--------------------------"
        "cars_data.csv:"
        "model: Марка автомобиля."
        "year: Год выпуска автомобиля."
        "motor_type: Тип двигателя (например, бензин, дизель и т. д.)."
        "running: Пробег автомобиля."
        "wheel: Расположение руля (например, слева, справа)."
        "color: Цвет автомобиля."
        "type: Тип кузова (например, седан, внедорожник и т. д.)."
        "status: Состояние автомобиля (например, отличное, хорошее и т. д.)."
        "motor_volume: Объем двигателя (например, в литрах)."
        "price: Цена автомобиля."
    elif (page=="Предикт"):
        st.title("Datasets")
        uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")
        if uploaded_file is not None:
            if uploaded_file.name == "apple2.csv":
                st.write("Файл apple2.csv был загружен")
                y = data_classification["Quality"]
                X = data_classification.drop(["Quality"], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
                model_classification = pickle.load(open('log_reg.pkl', 'rb'))
                predictions_classification = model_classification.predict(X_test)
                accuracy_classification = accuracy_score(y_test, predictions_classification)
                st.success(f"Точность: {accuracy_classification}")

            elif uploaded_file.name == "cars_data52.csv":
                st.write("Файл cars_data52.csv был загружен")
                model_regression = pickle.load(open('ridge.pkl', 'rb'))
                data_regression = pd.read_csv('cars_data52.csv')
                y = data_regression["price"]
                X = data_regression.drop(["price"], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
                predictions_regression = model_regression.predict(X_test)
                r2_score_regression = r2_score(y_test, predictions_regression)
                st.success(f"Коэффициент детерминации (R²): {r2_score_regression}")


            else:
                st.write("Загружен файл неизвестного формата")

if __name__ == "__main__":
    machine_learning()
