import pandas as pd


def load_data(path):
    return pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')


def clean_data(df):
    df = df.dropna()
    return df


def encode_data(df):
    df = pd.get_dummies(df, columns=["Plataforma", "Genero", "Editorial"])
    return df


def split_features(df):
    X = df.drop(["Ventas Global", "Nombre"], axis=1)
    y = df["Ventas Global"]
    return X, y