import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# MODELOS
def sigmoid(z): return 1 / (1 + np.exp(-z))

class LinearRegressionGD:
    def __init__(self, theta): self.theta_ = theta
    def predict(self, X): return np.c_[np.ones((X.shape[0], 1)), X] @ self.theta_

class LogisticRegressionOVR:
    def __init__(self, thetas, classes):
        self.thetas_, self.classes_ = thetas, classes
    def predict(self, X):
        probs = sigmoid(np.c_[np.ones((X.shape[0], 1)), X] @ self.thetas_.T)
        return self.classes_[np.argmax(probs, axis=1)]

# CARGA DE PESOS
mush_w = np.load("weights/mushroom_theta.npz", allow_pickle=True)
wine_w = np.load("weights/wine_theta.npz",     allow_pickle=True)

# Escaladores
scaler_mush = StandardScaler()
scaler_mush.mean_, scaler_mush.scale_ = mush_w["scaler_mean"], mush_w["scaler_scale"]

scaler_wine = StandardScaler()
scaler_wine.mean_, scaler_wine.scale_ = wine_w["scaler_mean"], wine_w["scaler_scale"]

# Modelos
log_model = LogisticRegressionOVR(mush_w["thetas"], mush_w["classes"])
lin_model = LinearRegressionGD(wine_w["theta"])

# LISTAS
mush_features = [
    "cap-shape","cap-surface","cap-color","bruises","odor",
    "gill-attachment","gill-spacing","gill-size","gill-color",
    "stalk-shape","stalk-root","stalk-surface-above-ring",
    "stalk-surface-below-ring","stalk-color-above-ring",
    "stalk-color-below-ring","veil-type","veil-color","ring-number",
    "ring-type","spore-print-color","population","habitat"
]
enc_categories = mush_w["enc_categories"].tolist()
mush_options = {f: list(cats) for f, cats in zip(mush_features, enc_categories)}

wine_features = [
    "fixed acidity","volatile acidity","citric acid","residual sugar",
    "chlorides","free sulfur dioxide","total sulfur dioxide","density",
    "pH","sulphates","alcohol"
]
wine_ranges = {
    "fixed acidity": (4.0, 16.0),
    "volatile acidity": (0.10, 1.60),
    "citric acid": (0.0, 1.0),
    "residual sugar": (0.6, 15.0),
    "chlorides": (0.01, 0.3),
    "free sulfur dioxide": (1, 72),
    "total sulfur dioxide": (6, 289),
    "density": (0.9900, 1.0040),
    "pH": (2.8, 4.2),
    "sulphates": (0.3, 2.0),
    "alcohol": (8.0, 15.5)
}

# ONE-HOT
def one_hot_mush(row_dict):
    vec = []
    for feat, cats in zip(mush_features, enc_categories):
        chosen = row_dict[feat]
        vec.extend([1.0 if chosen == cat else 0.0 for cat in cats])
    return np.array([vec], dtype=float)

# FUNCIONES DE PREDICCION
def predict_single_mushroom(*vals):
    row = dict(zip(mush_features, vals))
    x_onehot = one_hot_mush(row)
    x_scaled = scaler_mush.transform(x_onehot)
    pred = log_model.predict(x_scaled)[0]
    label = "Comestible" if pred == 0 else "Venenoso"
    return f" Predicción: **{label}**"

def predict_single_wine(*vals):
    data = dict(zip(wine_features, vals))
    x_scaled = scaler_wine.transform([list(data.values())])
    y_hat = lin_model.predict(x_scaled)[0]
    return f" Calidad estimada: **{round(y_hat, 2)} / 10**"

# INTERFAZ
with gr.Blocks(title="Proyecto AN 2025") as demo:
    gr.Markdown("## 🔮 Interfaz interactiva — Modelos desde cero")

    with gr.Tabs():
        # -TB HONGOS
        with gr.Tab("Clasificación de Hongos"):
            gr.Markdown(
                "Selecciona las características del hongo y pulsa **Predecir**."
            )

            mush_inputs = {}
            # Distribuir 11 y 11 en dos columnas
            with gr.Row():
                with gr.Column():
                    for feat in mush_features[:11]:
                        mush_inputs[feat] = gr.Dropdown(
                            mush_options[feat],
                            value=mush_options[feat][0],
                            label=feat.replace("-", " ").title()
                        )
                with gr.Column():
                    for feat in mush_features[11:]:
                        mush_inputs[feat] = gr.Dropdown(
                            mush_options[feat],
                            value=mush_options[feat][0],
                            label=feat.replace("-", " ").title()
                        )

            out_mush = gr.Markdown()
            gr.Button("Predecir").click(
                fn=predict_single_mushroom,
                inputs=list(mush_inputs.values()),
                outputs=out_mush
            )

        # TB VINO
        with gr.Tab("Calidad del Vino"):
            gr.Markdown("Ajusta los valores físico-químicos y pulsa **Predecir**.")

            wine_inputs = {}
            with gr.Row():
                with gr.Column():
                    for feat in wine_features[:6]:
                        mn, mx = wine_ranges[feat]
                        wine_inputs[feat] = gr.Slider(
                            mn, mx, step=(mx - mn) / 200,
                            value=(mn + mx) / 2, label=feat.title()
                        )
                with gr.Column():
                    for feat in wine_features[6:]:
                        mn, mx = wine_ranges[feat]
                        wine_inputs[feat] = gr.Slider(
                            mn, mx, step=(mx - mn) / 200,
                            value=(mn + mx) / 2, label=feat.title()
                        )

            out_wine = gr.Markdown()
            gr.Button("Predecir").click(
                fn=predict_single_wine,
                inputs=list(wine_inputs.values()),
                outputs=out_wine
            )

if __name__ == "__main__":
    demo.launch()