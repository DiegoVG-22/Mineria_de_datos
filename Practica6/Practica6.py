import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# cargar datos
df = pd.read_csv("champions_league_matches.csv")

# limpiar datos
df["home_possession"] = df["home_possession"].str.replace("%","").astype(float)
df["away_possession"] = df["away_possession"].str.replace("%","").astype(float)

df = df.dropna()

# variables (X) y etiqueta (y)
X = df[[
    "home_possession",
    "away_possession",
    "home_shots_on_target_pct",
    "away_shots_on_target_pct"
]]

y = df["result"]

# entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# predicciones
y_pred = knn.predict(X_test)

# evaluacion
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))