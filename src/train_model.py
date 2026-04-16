import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import load_data, clean_data, encode_data, split_features

# ==============================
# CARGA Y PREPARACIÓN DE DATOS
# ==============================

df = load_data("data/ventas_videojuegos.csv")
df = clean_data(df)
df = encode_data(df)

X, y = split_features(df)

# ❗ ELIMINAR DATA LEAKAGE
X = X.drop(["Ventas NA", "Ventas EU", "Ventas JP", "Ventas Otros"], axis=1)

# ==============================
# SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODELO RANDOM FOREST
# ==============================

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(" RMSE Random Forest:", rmse)
print(" R2 Random Forest:", r2)

# ==============================
# MODELO LINEAR REGRESSION
# ==============================

lr = LinearRegression()
lr.fit(X_train, y_train)

preds_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, preds_lr))

print("RMSE Linear Regression :", rmse_lr)

# ==============================
# GUARDAR MODELO
# ==============================

joblib.dump(model, "models/modelo_rf.pkl")

# ==============================
# FEATURE IMPORTANCE
# ==============================

importancias = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importancias = importancias.sort_values(by="importance", ascending=False)

print(importancias.head(10))

# ==============================
# GRÁFICO
# ==============================

importancias.head(10).plot(kind='barh', x='feature', y='importance')
plt.gca().invert_yaxis()
plt.title("Top 10 Variables más importantes")
plt.show()