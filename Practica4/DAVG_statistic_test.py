import pandas as pd
from scipy.stats import f_oneway, ttest_ind, kruskal

# cargar dataset
df = pd.read_csv("champions_league_matches.csv")

# limpiar datos
df["home_possession"] = df["home_possession"].str.replace("%","").astype(float)
df["away_possession"] = df["away_possession"].str.replace("%","").astype(float)

# eliminar nulos
df = df.dropna(subset=["home_possession", "away_possession", "result"])

# ANOVA
home_win = df[df["result"] == "Home Win"]["home_possession"]
away_win = df[df["result"] == "Away Win"]["home_possession"]
draw = df[df["result"] == "Draw"]["home_possession"]

f_stat, p_value = f_oneway(home_win, away_win, draw)

print("Resultados ANOVA")
print("F-statistic:", f_stat)
print("P-value:", p_value)

# T-TEST (solo 2 grupos)
t_stat, p_value_t = ttest_ind(home_win, away_win, nan_policy='omit')

print("\nT-TEST (Victoria local vs Victoria visitante)")
print("T-statistic:", t_stat)
print("P-value:", p_value_t)

# KRUSKAL-WALLIS
k_stat, p_value_k = kruskal(home_win, away_win, draw)

print("\nKRUSKAL-WALLIS")
print("Statistic:", k_stat)
print("P-value:", p_value_k)