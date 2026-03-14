import pandas as pd
import matplotlib.pyplot as plt

# cargar dataset
df = pd.read_csv("champions_league_matches.csv")

# limpiar columnas de porcentaje
df["home_possession"] = df["home_possession"].str.replace("%", "").astype(float)
df["away_possession"] = df["away_possession"].str.replace("%", "").astype(float)

# convertir columnas numéricas
df["home_shots_on_target"] = pd.to_numeric(df["home_shots_on_target"], errors="coerce")
df["away_shots_on_target"] = pd.to_numeric(df["away_shots_on_target"], errors="coerce")
df["home_saves"] = pd.to_numeric(df["home_saves"], errors="coerce")
df["away_saves"] = pd.to_numeric(df["away_saves"], errors="coerce")

# Gráfica de pastel - resultados
resultados = df["result"].value_counts()

plt.figure()
resultados.plot.pie(autopct='%1.1f%%')
plt.title("Distribución de resultados de los partidos")
plt.ylabel("")
plt.show()

# Boxplot - posesión del balón
df[["home_possession", "away_possession"]].plot.box()
plt.title("Distribución de posesión del balón")
plt.ylabel("Porcentaje de posesión")
plt.show()

# Scatter Plot
plt.figure()
plt.scatter(df["home_possession"], df["away_possession"])
plt.title("Posesión del equipo local vs visitante")
plt.xlabel("Posesión equipo local (%)")
plt.ylabel("Posesión equipo visitante (%)")
plt.show()

# Gráfica de barras - promedio por equipo
promedio_posesion = df.groupby("home_team")["home_possession"].mean()

plt.figure(figsize=(10,5))
promedio_posesion.sort_values().plot.bar()
plt.title("Promedio de posesión por equipo local")
plt.ylabel("Promedio de posesión (%)")
plt.xlabel("Equipos")
plt.show()

# Histogramas con for loop
columnas = [
    "home_possession",
    "away_possession",
    "home_shots_on_target_pct",
    "away_shots_on_target_pct"
]

for col in columnas:
    plt.figure()
    df[col].hist(bins=10)
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.show()