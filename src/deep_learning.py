# Deep Learning Model for Video Game Sales Prediction
# Includes normalization, early stopping and TensorBoard visualization

import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

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
# SPLIT (REPRODUCIBLE)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# NORMALIZACIÓN
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# MODELO
# ==============================

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# ==============================
# CALLBACKS
# ==============================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

tensorboard = TensorBoard(log_dir='logs/deep_learning')

# ==============================
# ENTRENAMIENTO
# ==============================

print("Entrenando modelo...")

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard, early_stop],
    verbose=1
)

print("Modelo entrenado!")

# ==============================
# GUARDAR MODELO
# ==============================

model.save("models/modelo_dl.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("Modelo y scaler guardados correctamente.")