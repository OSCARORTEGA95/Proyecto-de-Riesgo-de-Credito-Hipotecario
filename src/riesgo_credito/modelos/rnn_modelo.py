"""Red recurrente (GRU) en NumPy, compatible con scikit-learn, y transformadores que arman sus secuencias.

La red lee una secuencia de pasos con una celda GRU (compuertas de actualización z y de
reinicio r), toma el último estado oculto, le pega variables estáticas opcionales y lo pasa
por capas densas ReLU con dropout hasta una salida sigmoide. Se entrena con Adam, recorte
de gradiente y parada temprana, igual que la red densa de red_modelo.py, y cabe en el mismo
Pipeline, la misma corrección de probabilidad y el mismo empaquetado de MLflow.

Formato de entrada (una fila por ejemplo, plana para que encaje en scikit-learn):
    [ paso_1 (n_entrada) | ... | paso_T (n_entrada) | máscara (T) | estáticas (S) ]
La máscara vale 1 si el paso existe y 0 si no (por ejemplo, un mes que el crédito no tuvo):
en un paso con máscara 0 el estado oculto pasa igual al siguiente.

Celda GRU, paso t:
    z = σ(x Wz + h Uz + bz)          compuerta de actualización
    r = σ(x Wr + h Ur + br)          compuerta de reinicio
    n = tanh(x Wn + (r ⊙ h) Un + bn) estado candidato
    h' = (1 − z) ⊙ n + z ⊙ h
"""
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import roc_auc_score


# Los mismos valores de knn_modelo.preparar; se repiten aquí para no depender de mlflow al predecir
MESES_SIN_ATRASO = 120
LIMITE_DIF_TASA = 5.0


def _sigmoide(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class RedRecurrente(ClassifierMixin, BaseEstimator):
    def __init__(self, n_pasos=12, n_entrada=1, unidades=32, densas=(32,), dropout=0.0, epocas=30, tam_lote=256,
                 tasa_aprendizaje=2e-3, l2=1e-5, recorte=5.0, paciencia=None, semilla=42, verbose=False):
        self.n_pasos = n_pasos
        self.n_entrada = n_entrada
        self.unidades = unidades
        self.densas = densas
        self.dropout = dropout
        self.epocas = epocas
        self.tam_lote = tam_lote
        self.tasa_aprendizaje = tasa_aprendizaje
        self.l2 = l2
        self.recorte = recorte
        self.paciencia = paciencia
        self.semilla = semilla
        self.verbose = verbose

    # ---------- partes de la entrada ----------

    def _partes(self, X):
        T, d = int(self.n_pasos), int(self.n_entrada)
        secuencia = X[:, :T * d].reshape(len(X), T, d)
        mascara = X[:, T * d:T * d + T]
        estaticas = X[:, T * d + T:]
        return secuencia, mascara, estaticas

    # ---------- parámetros ----------

    def _iniciar(self, n_estaticas, rng):
        d, H = int(self.n_entrada), int(self.unidades)
        lim = np.sqrt(6.0 / (d + H))
        self.W_ = rng.uniform(-lim, lim, (d, 3 * H)).astype(np.float32)
        self.U_ = np.concatenate([np.linalg.qr(rng.standard_normal((H, H)))[0] for _ in range(3)], axis=1).astype(np.float32)
        self.b_ = np.zeros(3 * H, dtype=np.float32)
        tamanos = [H + n_estaticas, *[int(c) for c in self.densas], 1]
        self.pesos_, self.sesgos_ = [], []
        for a, b in zip(tamanos[:-1], tamanos[1:]):
            self.pesos_.append((rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32))
            self.sesgos_.append(np.zeros(b, dtype=np.float32))
        self._m = [np.zeros_like(p) for p in self._parametros()]
        self._v = [np.zeros_like(p) for p in self._parametros()]
        self._t = 0

    def _parametros(self):
        return [self.W_, self.U_, self.b_] + self.pesos_ + self.sesgos_

    # ---------- propagación ----------

    def _adelante(self, X, rng=None):
        secuencia, mascara, estaticas = self._partes(X)
        H = int(self.unidades)
        h = np.zeros((len(X), H), dtype=np.float32)
        pasos = []
        for t in range(int(self.n_pasos)):
            x, m = secuencia[:, t, :], mascara[:, t:t + 1]
            a = x @ self.W_ + self.b_
            zr = _sigmoide(a[:, :2 * H] + h @ self.U_[:, :2 * H])
            z, r = zr[:, :H], zr[:, H:]
            rh = r * h
            n = np.tanh(a[:, 2 * H:] + rh @ self.U_[:, 2 * H:])
            h_nuevo = (1.0 - z) * n + z * h
            pasos.append((x, h, z, r, rh, n, m))
            h = m * h_nuevo + (1.0 - m) * h
        activaciones, mascaras = [np.concatenate([h, estaticas], axis=1)], []
        a = activaciones[0]
        ultima = len(self.pesos_) - 1
        for i, (W, b) in enumerate(zip(self.pesos_, self.sesgos_)):
            zc = a @ W + b
            if i == ultima:
                activaciones.append(zc)
                break
            a = np.maximum(zc, 0.0)
            if rng is not None and self.dropout > 0:
                mk = (rng.random(a.shape, dtype=np.float32) >= self.dropout).astype(np.float32) / (1.0 - self.dropout)
                a = a * mk
                mascaras.append(mk)
            else:
                mascaras.append(None)
            activaciones.append(a)
        return pasos, activaciones, mascaras

    def _logit(self, X, lote=20000):
        X = np.asarray(X, dtype=np.float32)
        salida = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), lote):
            salida[i:i + lote] = self._adelante(X[i:i + lote])[1][-1][:, 0]
        return salida

    def _gradientes(self, X, y, rng):
        """Pérdida (entropía cruzada) y gradientes de todos los parámetros, en el orden de _parametros()."""
        n, H = len(X), int(self.unidades)
        pasos, activaciones, mascaras = self._adelante(X, rng)
        logit = activaciones[-1][:, 0]
        p = _sigmoide(logit)
        perdida = float(np.mean(np.logaddexp(0.0, logit) - y * logit))
        delta = ((p - y) / n).astype(np.float32)[:, None]
        gw, gb = [None] * len(self.pesos_), [None] * len(self.pesos_)
        for i in range(len(self.pesos_) - 1, -1, -1):
            gw[i] = activaciones[i].T @ delta + self.l2 * self.pesos_[i]
            gb[i] = delta.sum(axis=0)
            delta = delta @ self.pesos_[i].T
            if i > 0:
                delta = delta * (activaciones[i] > 0)
                if mascaras[i - 1] is not None:
                    delta = delta * mascaras[i - 1]
        dh = delta[:, :H]  # gradiente sobre el último estado oculto (las estáticas no tienen parámetros propios)
        dW, dU, db = self.l2 * self.W_, self.l2 * self.U_, np.zeros_like(self.b_)
        dW, dU = dW.copy(), dU.copy()
        for x, h_prev, z, r, rh, nc, m in reversed(pasos):
            dh_nuevo = dh * m
            dh_prev = dh * (1.0 - m) + dh_nuevo * z
            dz = dh_nuevo * (h_prev - nc) * z * (1.0 - z)
            dan = dh_nuevo * (1.0 - z) * (1.0 - nc * nc)
            dU[:, 2 * H:] += rh.T @ dan
            drh = dan @ self.U_[:, 2 * H:].T
            dh_prev += drh * r
            dr = drh * h_prev * r * (1.0 - r)
            dzr = np.concatenate([dz, dr], axis=1)
            dU[:, :2 * H] += h_prev.T @ dzr
            dh_prev += dzr @ self.U_[:, :2 * H].T
            da = np.concatenate([dz, dr, dan], axis=1)
            dW += x.T @ da
            db += da.sum(axis=0)
            dh = dh_prev
        return perdida, [dW, dU, db] + gw + gb

    def _paso(self, X, y, rng):
        perdida, grads = self._gradientes(X, y, rng)
        norma = np.sqrt(sum(float((g * g).sum()) for g in grads))
        if self.recorte and norma > self.recorte:
            grads = [g * (self.recorte / norma) for g in grads]
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        tasa = self.tasa_aprendizaje * np.sqrt(1 - b2 ** self._t) / (1 - b1 ** self._t)
        for j, (par, g) in enumerate(zip(self._parametros(), grads)):
            self._m[j] = b1 * self._m[j] + (1 - b1) * g
            self._v[j] = b2 * self._v[j] + (1 - b2) * g * g
            par -= (tasa * self._m[j] / (np.sqrt(self._v[j]) + eps)).astype(np.float32)
        return perdida

    # ---------- entrenamiento y predicción ----------

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        rng = np.random.default_rng(self.semilla)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        n_estaticas = X.shape[1] - int(self.n_pasos) * (int(self.n_entrada) + 1)
        if n_estaticas < 0:
            raise ValueError(f"La entrada tiene {X.shape[1]} columnas; se esperaban al menos "
                             f"{int(self.n_pasos) * (int(self.n_entrada) + 1)} (pasos × entrada + máscara)")
        self.n_estaticas_ = n_estaticas
        self._iniciar(n_estaticas, rng)
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
                    mejores = [p.copy() for p in self._parametros()]
                else:
                    sin_mejora += 1
                    if self.paciencia and sin_mejora >= self.paciencia:
                        break
        if mejores is not None:
            self.W_, self.U_, self.b_ = mejores[:3]
            k = len(self.pesos_)
            self.pesos_, self.sesgos_ = mejores[3:3 + k], mejores[3 + k:]
        self.mejor_epoca_ = mejor_epoca or len(self.historial_)
        self.mejor_auc_validacion_ = mejor_auc if mejores is not None else None
        del self._m, self._v
        return self

    def predict_proba(self, X):
        p = _sigmoide(self._logit(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_parametros(self):
        return int(sum(p.size for p in [self.W_, self.U_, self.b_] + self.pesos_ + self.sesgos_))


# ---------- secuencias de la pregunta 1: columnas de la solicitud ----------

class SecuenciaDeColumnas(TransformerMixin, BaseEstimator):
    """Convierte las D columnas ya codificadas de la solicitud en D pasos.

    El paso k lleva [valor de la columna k, indicador one-hot de la columna k]: la red lee
    la solicitud columna por columna y sabe cuál está leyendo. Las variables de la solicitud
    no tienen orden en el tiempo; esta versión existe para comparar con la red densa.
    """

    def fit(self, X, y=None):
        self.n_columnas_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        n, D = X.shape
        pasos = np.zeros((n, D, D + 1), dtype=np.float32)
        pasos[:, :, 0] = X
        pasos[:, np.arange(D), 1 + np.arange(D)] = 1.0
        return np.concatenate([pasos.reshape(n, -1), np.ones((n, D), dtype=np.float32)], axis=1)


# ---------- secuencias de la pregunta 2: historia mensual del crédito ----------

def nombre_mes(variable, rezago):
    """Columna de la variable `rezago` meses antes del mes que se predice (0 = el mes actual)."""
    return f"{variable}__m{rezago}"


def indice_mes(period):
    period = np.asarray(period, dtype=np.int64)
    return (period // 100) * 12 + (period % 100) - 1


def construir_ancho(objetivos, historia, variables_mes, n_pasos):
    """Une a cada fila objetivo (crédito, mes t) sus `n_pasos` meses de historia, en columnas anchas.

    `historia` tiene una fila por crédito-mes con loan_id, period y las variables mensuales.
    Un mes ausente (antes del primer pago, o con más de 30 días de atraso, que la tabla de la
    pregunta 2 no guarda) queda vacío y con presente = 0.
    """
    import pandas as pd
    h = historia[["loan_id", "period"] + variables_mes].copy()
    h["_idx"] = indice_mes(h["period"])
    h["_existe"] = True
    h = h.drop(columns="period").drop_duplicates(["loan_id", "_idx"]).set_index(["loan_id", "_idx"])
    base = objetivos.reset_index(drop=True)
    idx = indice_mes(base["period"])
    columnas = {}
    for rezago in range(n_pasos):
        claves = pd.MultiIndex.from_arrays([base["loan_id"].to_numpy(), idx - rezago])
        filas = h.reindex(claves)
        presente = filas["_existe"].fillna(False).to_numpy(dtype=bool)
        for v in variables_mes:
            columnas[nombre_mes(v, rezago)] = filas[v].to_numpy()
        columnas[nombre_mes("presente", rezago)] = presente
    return pd.concat([base, pd.DataFrame(columnas, index=base.index)], axis=1)


class SecuenciaMensual(TransformerMixin, BaseEstimator):
    """Toma las columnas anchas de construir_ancho (ya pasadas por preparar) y arma el formato de RedRecurrente.

    Las variables mensuales se imputan con la mediana y se estandarizan con la media y la
    desviación de todos los meses presentes del entrenamiento; los meses ausentes quedan en 0
    con máscara 0. Las estáticas pasan por su propio preprocesador (opcional).
    """

    def __init__(self, variables_mes, n_pasos=12, estaticas=None):
        self.variables_mes = variables_mes
        self.n_pasos = n_pasos
        self.estaticas = estaticas

    def _bloque(self, X):
        import pandas as pd
        T, V = int(self.n_pasos), len(self.variables_mes)
        valores = np.empty((len(X), T, V), dtype=np.float64)
        mascara = np.empty((len(X), T), dtype=np.float32)
        for t in range(T):
            rezago = T - 1 - t  # el paso 0 es el mes más antiguo y el último es el mes actual
            mascara[:, t] = np.asarray(X[nombre_mes("presente", rezago)], dtype=np.float32)
            for j, v in enumerate(self.variables_mes):
                col = pd.to_numeric(X[nombre_mes(v, rezago)], errors="coerce").to_numpy(dtype=np.float64)
                if v == "meses_desde_atraso":
                    col = np.where(np.isnan(col), MESES_SIN_ATRASO, col)
                elif v == "dif_tasa":
                    col = np.clip(col, -LIMITE_DIF_TASA, LIMITE_DIF_TASA)
                valores[:, t, j] = col
        return valores, mascara

    def fit(self, X, y=None):
        valores, mascara = self._bloque(X)
        presentes = valores[mascara > 0]
        self.mediana_ = np.nanmedian(presentes, axis=0)
        self.mediana_ = np.where(np.isnan(self.mediana_), 0.0, self.mediana_)
        llenos = np.where(np.isnan(presentes), self.mediana_, presentes)
        self.media_ = llenos.mean(axis=0)
        self.desviacion_ = llenos.std(axis=0)
        self.desviacion_[self.desviacion_ < 1e-9] = 1.0
        if self.estaticas is not None:
            self.estaticas.fit(X)
        return self

    def transform(self, X):
        valores, mascara = self._bloque(X)
        valores = np.where(np.isnan(valores), self.mediana_, valores)
        valores = (valores - self.media_) / self.desviacion_
        valores = valores * mascara[:, :, None]
        partes = [valores.reshape(len(X), -1).astype(np.float32), mascara]
        if self.estaticas is not None:
            partes.append(np.asarray(self.estaticas.transform(X), dtype=np.float32))
        return np.concatenate(partes, axis=1)
