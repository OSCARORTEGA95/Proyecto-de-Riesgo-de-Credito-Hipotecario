"""Red neuronal densa (perceptrón multicapa) con dropout, compatible con scikit-learn.

scikit-learn no ofrece dropout en MLPClassifier, así que la red se implementa aquí con NumPy:
capas ocultas ReLU, salida sigmoide, pérdida de entropía cruzada, optimizador Adam, dropout
invertido en las capas ocultas y regularización L2. Al ser un estimador de scikit-learn con
predict_proba, entra en el mismo Pipeline, en la misma corrección de la probabilidad y en el
mismo empaquetado de MLflow que los modelos de k vecinos, la regresión logística y naive Bayes,
sin depender de TensorFlow ni de PyTorch al predecir.

fit guarda en historial_ una fila por época con la pérdida de entrenamiento y, si se le pasa
un conjunto de validación, el área bajo la curva en validación. Con paciencia, se detiene
cuando la validación deja de mejorar y conserva los pesos de la mejor época.
"""
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score


def _sigmoide(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # estable para valores grandes


class RedNeuronal(ClassifierMixin, BaseEstimator):
    def __init__(self, capas=(64, 32), dropout=0.0, epocas=30, tam_lote=512, tasa_aprendizaje=1e-3,
                 l2=1e-5, paciencia=None, semilla=42, verbose=False):
        self.capas = capas
        self.dropout = dropout
        self.epocas = epocas
        self.tam_lote = tam_lote
        self.tasa_aprendizaje = tasa_aprendizaje
        self.l2 = l2
        self.paciencia = paciencia
        self.semilla = semilla
        self.verbose = verbose

    # ---------- propagación ----------

    def _adelante(self, X, rng=None):
        """Devuelve las activaciones de cada capa y las máscaras de dropout (solo si rng, en entrenamiento)."""
        activaciones, mascaras = [X], []
        a = X
        ultima = len(self.pesos_) - 1
        for i, (W, b) in enumerate(zip(self.pesos_, self.sesgos_)):
            z = a @ W + b
            if i == ultima:
                activaciones.append(z)  # logit de la salida
                break
            a = np.maximum(z, 0.0)
            if rng is not None and self.dropout > 0:
                m = (rng.random(a.shape, dtype=np.float32) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
                a = a * m
                mascaras.append(m)
            else:
                mascaras.append(None)
            activaciones.append(a)
        return activaciones, mascaras

    def _logit(self, X, lote=20000):
        X = np.asarray(X, dtype=np.float32)
        salida = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), lote):
            salida[i:i + lote] = self._adelante(X[i:i + lote])[0][-1][:, 0]
        return salida

    # ---------- entrenamiento ----------

    def _iniciar(self, n_entradas, rng):
        tamanos = [n_entradas, *[int(c) for c in self.capas], 1]
        self.pesos_, self.sesgos_ = [], []
        for a, b in zip(tamanos[:-1], tamanos[1:]):
            self.pesos_.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))  # He
            self.sesgos_.append(np.zeros(b, dtype=np.float32))
        self._m = [np.zeros_like(p) for p in self.pesos_ + self.sesgos_]
        self._v = [np.zeros_like(p) for p in self.pesos_ + self.sesgos_]
        self._t = 0

    def _paso(self, X, y, rng):
        n = len(X)
        activaciones, mascaras = self._adelante(X, rng)
        logit = activaciones[-1][:, 0]
        p = _sigmoide(logit)
        perdida = np.mean(np.logaddexp(0.0, logit) - y * logit)  # entropía cruzada estable
        delta = ((p - y) / n).astype(np.float32)[:, None]
        grad_w, grad_b = [None] * len(self.pesos_), [None] * len(self.pesos_)
        for i in range(len(self.pesos_) - 1, -1, -1):
            grad_w[i] = activaciones[i].T @ delta + self.l2 * self.pesos_[i]
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.pesos_[i].T
                delta = delta * (activaciones[i] > 0)
                if mascaras[i - 1] is not None:
                    delta = delta * mascaras[i - 1]
        # Adam
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        tasa = self.tasa_aprendizaje * np.sqrt(1 - b2 ** self._t) / (1 - b1 ** self._t)
        parametros = self.pesos_ + self.sesgos_
        for j, (par, g) in enumerate(zip(parametros, grad_w + grad_b)):
            self._m[j] = b1 * self._m[j] + (1 - b1) * g
            self._v[j] = b2 * self._v[j] + (1 - b2) * g * g
            par -= (tasa * self._m[j] / (np.sqrt(self._v[j]) + eps)).astype(np.float32)
        return float(perdida)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        rng = np.random.default_rng(self.semilla)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self._iniciar(X.shape[1], rng)
        self.historial_ = []
        mejor_auc, mejor_epoca, mejores, sin_mejora = -np.inf, 0, None, 0
        for epoca in range(1, int(self.epocas) + 1):
            t0 = time.time()
            orden = rng.permutation(len(X))
            perdidas, pesos_lote = [], []
            for i in range(0, len(X), self.tam_lote):
                idx = orden[i:i + self.tam_lote]
                perdidas.append(self._paso(X[idx], y[idx], rng))
                pesos_lote.append(len(idx))
            fila = {"epoca": epoca, "perdida": float(np.average(perdidas, weights=pesos_lote))}
            if X_val is not None:
                logit_val = self._logit(X_val)
                yv = np.asarray(y_val, dtype=float)
                fila["perdida_validacion"] = float(np.mean(np.logaddexp(0.0, logit_val) - yv * logit_val))
                fila["auc_validacion"] = float(roc_auc_score(yv, logit_val))
            fila["segundos"] = time.time() - t0
            self.historial_.append(fila)
            if self.verbose:
                print(f"  época {epoca}: " + ", ".join(f"{k} {v:.4f}" for k, v in fila.items() if k != "epoca"),
                      flush=True)
            if X_val is not None:
                if fila["auc_validacion"] > mejor_auc:
                    mejor_auc, mejor_epoca, sin_mejora = fila["auc_validacion"], epoca, 0
                    mejores = ([w.copy() for w in self.pesos_], [b.copy() for b in self.sesgos_])
                else:
                    sin_mejora += 1
                    if self.paciencia and sin_mejora >= self.paciencia:
                        break
        if mejores is not None:  # pesos de la mejor época en validación
            self.pesos_, self.sesgos_ = mejores
        self.mejor_epoca_ = mejor_epoca or len(self.historial_)
        self.mejor_auc_validacion_ = mejor_auc if mejores is not None else None
        del self._m, self._v  # el estado del optimizador no hace falta para predecir
        return self

    # ---------- predicción ----------

    def predict_proba(self, X):
        p = _sigmoide(self._logit(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_parametros(self):
        return int(sum(w.size + b.size for w, b in zip(self.pesos_, self.sesgos_)))
