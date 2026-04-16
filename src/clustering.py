import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import load_data, clean_data, encode_data


df = load_data("data/ventas_videojuegos.csv")
df = clean_data(df)
df = encode_data(df)

X = df.drop(["Ventas Global", "Nombre"], axis=1)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

print(df["Cluster"].value_counts())