import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# cargar dataset
df = pd.read_csv("champions_league_matches.csv")

# limpiar fecha
df["date"] = pd.to_datetime(df["date"])

# limpiar posesion
df["home_possession"] = (
    df["home_possession"]
    .str.replace("%", "")
    .astype(float)
)

# eliminar nulos
df = df.dropna(subset=["date", "home_possession"])

# convertir fechas a numeros
df["date_ordinal"] = df["date"].map(pd.Timestamp.toordinal)

# variables
X = df[["date_ordinal"]]
y = df["home_possession"]

# modelo
model = LinearRegression()
model.fit(X, y)

# predicciones existentes
y_pred = model.predict(X)

# R2
r2 = r2_score(y, y_pred)

print("R2 score:", r2)

# predecir fechas nuevas

future_dates = pd.date_range(
    start=df["date"].max(),
    periods=5
)

future_ordinal = future_dates.map(pd.Timestamp.toordinal)

future_predictions = model.predict(
    future_ordinal.values.reshape(-1,1)
)

print("\nPredicciones a futuro:")

for date, pred in zip(future_dates, future_predictions):
    print(date.date(), "=", round(pred,2))

# grafica
plt.figure(figsize=(10,5))

plt.scatter(df["date"], y)

plt.plot(df["date"], y_pred)

plt.title("Forecasting de posesion local")
plt.xlabel("Fecha")
plt.ylabel("Posesion local (%Porcentaje)")

plt.show()