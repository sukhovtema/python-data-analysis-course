import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

with open("realty_data.csv", encoding="utf-8", errors="ignore") as f:
    df = pd.read_csv(f)

df = df[["total_square", "rooms", "floor", "price"]].dropna()

X = df[["total_square", "rooms", "floor"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

st.title("Прогнозирование стоимости недвижимости")

total_square = st.number_input("Общая площадь (м²)", min_value=1)
rooms = st.number_input("Количество комнат", min_value=1, max_value=10)
floor = st.number_input("Этаж", min_value=1)

if st.button("Прогнозировать цену"):
    input_data = np.array([[total_square, rooms, floor]])
    predicted_price = model.predict(input_data)[0]

    st.write(f"Прогнозируемая цена недвижимости: {predicted_price:.2f} рублей")

