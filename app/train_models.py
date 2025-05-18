import os
os.makedirs("weights", exist_ok=True)

import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from models import gradient_descent, LogisticRegressionOVR, LinearRegressionGD, sigmoid, ce_loss, mse_half

# ---------- MUSHROOM (clasificación) ----------
mush = pd.read_csv("mushrooms.csv")
X_cat = mush.drop("class", axis=1)
y     = mush["class"].map({"e":0, "p":1}).values
enc   = OneHotEncoder(sparse_output=False).fit(X_cat)
X_enc = enc.transform(X_cat)
scaler_m = StandardScaler().fit(X_enc)
X_scaled = scaler_m.transform(X_enc)

classes = np.unique(y)
thetas  = []
for cls in classes:
    y_bin = (y==cls).astype(int)
    theta = gradient_descent(
        X_scaled, y_bin, lr=0.1, epochs=5000,
        hypothesis=sigmoid, cost_fn=ce_loss)
    thetas.append(theta)
np.savez(
    "weights/mushroom_theta.npz",
    thetas=np.vstack(thetas),
    classes=classes,
    # 👇 convertir a array de objetos
    enc_categories=np.array(enc.categories_, dtype=object),
    scaler_mean=scaler_m.mean_,
    scaler_scale=scaler_m.scale_
)


# ---------- WINE (regresión) ----------
wine = pd.read_csv("winequality-red.csv", sep=";")
X, y = wine.drop("quality", axis=1).values, wine["quality"].values
scaler_w = StandardScaler().fit(X)
X_scaled = scaler_w.transform(X)
theta_w  = gradient_descent(
    X_scaled, y, lr=0.01, epochs=5000,
    hypothesis=lambda z: z, cost_fn=mse_half)
np.savez("weights/wine_theta.npz",
         theta=theta_w, scaler_mean=scaler_w.mean_, scaler_scale=scaler_w.scale_)
