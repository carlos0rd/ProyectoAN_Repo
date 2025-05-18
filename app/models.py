import numpy as np

# ---------- funciones auxiliares ----------
def sigmoid(z): return 1 / (1 + np.exp(-z))

def ce_loss(y_true, y_pred):
    m, eps = len(y_true), 1e-15
    return -(1/m)*np.sum(
        y_true*np.log(y_pred+eps)+(1-y_true)*np.log(1-y_pred+eps))

def mse_half(y_true, y_pred):
    return (1/(2*len(y_true)))*np.sum((y_pred - y_true)**2)

# ---------- descenso de gradiente ----------
def gradient_descent(X, y, theta=None, lr=0.01, epochs=1000,
                     hypothesis=lambda z: z, cost_fn=mse_half):
    m, n = X.shape
    X_b  = np.c_[np.ones((m,1)), X]
    theta = np.zeros(n+1) if theta is None else theta
    for _ in range(epochs):
        h = hypothesis(X_b @ theta)
        theta -= lr * (1/m) * X_b.T @ (h - y)
    return theta

# ---------- modelos ----------
class LinearRegressionGD:
    def __init__(self, theta): self.theta_ = theta
    def predict(self, X):
        return np.c_[np.ones((X.shape[0],1)), X] @ self.theta_

class LogisticRegressionOVR:
    def __init__(self, thetas, classes):
        self.thetas_, self.classes_ = thetas, classes
    def _proba(self, X):
        X_b = np.c_[np.ones((X.shape[0],1)), X]
        return sigmoid(X_b @ self.thetas_.T)
    def predict(self, X):
        return self.classes_[np.argmax(self._proba(X), axis=1)]
