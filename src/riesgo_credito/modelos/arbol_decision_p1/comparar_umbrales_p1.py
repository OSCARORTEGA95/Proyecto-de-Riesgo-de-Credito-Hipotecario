
import numpy as np

# Importar el modelo base.
# Al importarlo, se ejecuta nuevamente el entrenamiento.
from arbol_decision_p1 import modelo, X_val, y_val


# ==========================================
# 1. OBTENER PROBABILIDADES
# ==========================================

probabilidades = modelo.predict_proba(X_val)[:, 1]
reales = y_val.to_numpy().astype(int)

total = len(reales)
total_incumplimientos = int((reales == 1).sum())


# ==========================================
# 2. FUNCION PARA MOSTRAR RESULTADOS
# ==========================================

def evaluar_alertas(nombre, alertas):

    cantidad_alertas = int(alertas.sum())

    verdaderos_positivos = int(
        ((alertas == 1) & (reales == 1)).sum()
    )

    falsos_positivos = int(
        ((alertas == 1) & (reales == 0)).sum()
    )

    precision = (
        verdaderos_positivos / cantidad_alertas
        if cantidad_alertas > 0 else 0
    )

    recall = (
        verdaderos_positivos / total_incumplimientos
        if total_incumplimientos > 0 else 0
    )

    porcentaje_alertas = 100 * cantidad_alertas / total

    print(
        f"{nombre:<15} "
        f"{cantidad_alertas:>10,} "
        f"{porcentaje_alertas:>10.2f}% "
        f"{verdaderos_positivos:>8,} "
        f"{falsos_positivos:>10,} "
        f"{precision:>9.2%} "
        f"{recall:>9.2%}"
    )


# ==========================================
# 3. COMPARAR UMBRALES
# ==========================================

print("\nCOMPARACION DE UMBRALES - VALIDACION 2023")

print(
    f"{'Umbral':<15} "
    f"{'Alertas':>10} "
    f"{'% cartera':>11} "
    f"{'Aciertos':>8} "
    f"{'Falsas':>10} "
    f"{'Precision':>9} "
    f"{'Recall':>9}"
)

for umbral in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:

    alertas = probabilidades >= umbral

    evaluar_alertas(
        f"{umbral * 100:.1f}%",
        alertas
    )


# ==========================================
# 4. EVALUAR CREDITOS CON MAYOR RIESGO
# ==========================================

print("\nPRIORIZACION DE CREDITOS POR PROBABILIDAD")

for porcentaje in [1, 2, 5]:

    cantidad = max(
        1,
        int(total * porcentaje / 100)
    )

    posiciones = np.argpartition(
        probabilidades, -cantidad
    )[-cantidad:]

    alertas = np.zeros(total, dtype=bool)
    alertas[posiciones] = True

    evaluar_alertas(
        f"Top {porcentaje}%",
        alertas
    )

