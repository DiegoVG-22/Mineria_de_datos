import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# cargar dataset
df = pd.read_csv("champions_league_matches.csv")

# limpiar datos
df["home_possession"] = df["home_possession"].str.replace("%","").astype(float)
df["away_possession"] = df["away_possession"].str.replace("%","").astype(float)

df = df.dropna()

# variables para clustering
X = df[[
    "home_possession",
    "away_possession",
    "home_shots_on_target_pct",
    "away_shots_on_target_pct"
]]

# modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=42)

# entrenar modelo
kmeans.fit(X)

# clusters asignados
df["cluster"] = kmeans.labels_

print(df[["home_team", "away_team", "cluster"]].head())

# grafica
plt.figure(figsize=(8,6))

plt.scatter(
    df["home_possession"],
    df["away_possession"],
    c=df["cluster"]
)

plt.xlabel("Posesion Local")
plt.ylabel("Posesion Visitante")
plt.title("Clusters de partidos usando K-Means")

plt.show()