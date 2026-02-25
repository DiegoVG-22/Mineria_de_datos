import pandas as pd
import numpy as np

#Carga del DATASET
df = pd.read_csv("champions_league_matches.csv")

print("Primeras filas del dataset:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())


#Limpieza de datos

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["home_possession"] = df["home_possession"].str.replace("%", "", regex=False).astype(float)
df["away_possession"] = df["away_possession"].str.replace("%", "", regex=False).astype(float)

df["home_shots_on_target"] = pd.to_numeric(df["home_shots_on_target"], errors="coerce")
df["away_shots_on_target"] = pd.to_numeric(df["away_shots_on_target"], errors="coerce")

df["home_saves"] = pd.to_numeric(df["home_saves"], errors="coerce")
df["away_saves"] = pd.to_numeric(df["away_saves"], errors="coerce")


print("\nInformación del dataset:")
print(df.info())


# Estadísticas descriptivas generales


print("\nEstadísticas generales:")
print(df.describe())


# Estadísticas específicas


print("\n--- Promedios ---")
print("Promedio posesion local:", df["home_possession"].mean())
print("Promedio posesion visitante:", df["away_possession"].mean())

print("Promedio tiros a puerta local:", df["home_shots_on_target"].mean())
print("Promedio tiros a puerta visitante:", df["away_shots_on_target"].mean())

print("Promedio atajadas local:", df["home_saves"].mean())
print("Promedio atajadas visitante:", df["away_saves"].mean())


# Valores nulos


print("\nValores nulos por columna:")
print(df.isnull().sum())

# Estadísticas agrupadas por entidades


print("AGRUPACIONES POR ENTIDADES")

print("\nPromedio de tiros a puerta por equipo local:")
group_home_shots = df.groupby("home_team")["home_shots_on_target"].mean()
print(group_home_shots.sort_values(ascending=False))

print("\nPromedio de posesión por equipo local:")
group_home_possession = df.groupby("home_team")["home_possession"].mean()
print(group_home_possession.sort_values(ascending=False))

print("\nTotal de tiros a puerta por equipo visitante:")
group_away_shots = df.groupby("away_team")["away_shots_on_target"].sum()
print(group_away_shots.sort_values(ascending=False))

print("\nPromedio de tiros a puerta por estadio:")
group_venue = df.groupby("venue")["home_shots_on_target"].mean()
print(group_venue.sort_values(ascending=False))

print("\nCantidad de partidos dirigidos por árbitro:")
group_referee = df.groupby("referee")["home_team"].count()
print(group_referee.sort_values(ascending=False))