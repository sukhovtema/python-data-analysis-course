import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Загрузка данных
with open("realty_data.csv", encoding="utf-8", errors="ignore") as f:
    df = pd.read_csv(f)

# Подготовка данных
df = df[["total_square", "rooms", "floor", "price"]].dropna()
X = df[["total_square", "rooms", "floor"]]
y = df["price"]

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Сохраняем модель
joblib.dump(model, "model.pkl")