import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# cargar dataset
df = pd.read_csv("champions_league_matches.csv")

# combinar texto
text = " ".join(
    df["home_team"].dropna().astype(str)
) + " " + " ".join(
    df["away_team"].dropna().astype(str)
)

# crear word cloud
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

# mostrar grafica
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

plt.title("Nube de palabras de equipos")

plt.show()