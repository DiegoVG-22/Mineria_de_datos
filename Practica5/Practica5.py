import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# cargar datos
df = pd.read_csv("champions_league_matches.csv")

# limpiar datos
df["home_possession"] = df["home_possession"].str.replace("%","").astype(float)
df["away_possession"] = df["away_possession"].str.replace("%","").astype(float)

df = df.dropna(subset=["home_possession", "away_possession"])

# variables
X = df[["home_possession"]]
y = df["away_possession"]

# modelo
model = LinearRegression()
model.fit(X, y)

# predicciones
y_pred = model.predict(X)

# R2
r2 = r2_score(y, y_pred)
print("R2 score:", r2)

# grafica
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Modelo lineal: Posesion local vs visitante")
plt.xlabel("Posesion local (%)")
plt.ylabel("Posesion visitante (%)")
plt.show()