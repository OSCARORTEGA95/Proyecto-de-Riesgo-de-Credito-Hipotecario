"""Esquema de los archivos del Single-Family Loan-Level Dataset de Freddie Mac.

Define, para los dos formatos de publicación, el nombre de cada columna en su
posición, el tipo con que se lee y los códigos que significan "no disponible".

- Formato nuevo (desde julio de 2026): ``orig_AAAATn.txt`` (31 columnas) y
  ``perf_AAAATn.txt`` (35 columnas).
- Formato anterior: ``historical_data_AAAATn.txt`` (32 columnas) y
  ``historical_data_time_AAAATn.txt`` (32 columnas). Frente al formato nuevo,
  la originación trae además el administrador y el indicador de cancelación
  del seguro hipotecario, y no trae VantageScore; el desempeño no trae esos
  dos atributos ni los costos de reducción por quiebra.

El formato se detecta por el número de columnas de la primera fila, de modo
que el mismo código sirve para cualquier trimestre.

Las funciones solo leen: no escriben ni modifican archivos.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Columnas por formato, en el orden en que vienen en el archivo
# --------------------------------------------------------------------------

ORIGINACION_NUEVO = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
    "msa", "mi_pct", "num_units", "occupancy", "ocltv", "dti", "orig_upb", "oltv",
    "orig_rate", "channel", "ppm_flag", "amort_type", "state", "prop_type", "zip3",
    "loan_id", "purpose", "orig_term", "num_borrowers", "seller", "super_conforming",
    "pre_harp_id", "special_eligibility", "harp", "valuation_method", "io_flag",
    "vantage_score",
]

ORIGINACION_ANTERIOR = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
    "msa", "mi_pct", "num_units", "occupancy", "ocltv", "dti", "orig_upb", "oltv",
    "orig_rate", "channel", "ppm_flag", "amort_type", "state", "prop_type", "zip3",
    "loan_id", "purpose", "orig_term", "num_borrowers", "seller", "servicer",
    "super_conforming", "pre_harp_id", "special_eligibility", "harp",
    "valuation_method", "io_flag", "mi_cancel",
]

DESEMPENO_NUEVO = [
    "loan_id", "period", "current_upb", "dlq_status", "loan_age", "remaining_months",
    "defect_settlement_date", "mod_flag", "zero_balance_code", "zero_balance_date",
    "current_rate", "non_interest_upb", "ddlpi", "mi_recoveries", "net_sales_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_costs",
    "taxes_insurance", "misc_expenses", "actual_loss", "cum_mod_cost", "step_mod_flag",
    "payment_deferral", "eltv", "zero_balance_upb", "dlq_accrued_interest",
    "dlq_disaster", "assistance_plan", "current_mod_cost", "interest_bearing_upb",
    "mi_cancel", "servicer", "cramdown_costs",
]

DESEMPENO_ANTERIOR = DESEMPENO_NUEVO[:32]

ESQUEMAS = {
    ("originacion", 31): ("nuevo", ORIGINACION_NUEVO),
    ("originacion", 32): ("anterior", ORIGINACION_ANTERIOR),
    ("desempeno", 35): ("nuevo", DESEMPENO_NUEVO),
    ("desempeno", 32): ("anterior", DESEMPENO_ANTERIOR),
}

# --------------------------------------------------------------------------
# Tipos de lectura (las columnas no listadas se leen como texto)
# --------------------------------------------------------------------------

ENTEROS = {
    "credit_score", "first_payment_date", "maturity_date", "mi_pct", "num_units",
    "ocltv", "dti", "oltv", "orig_term", "num_borrowers", "valuation_method",
    "vantage_score", "period", "loan_age", "remaining_months",
    "defect_settlement_date", "zero_balance_date", "ddlpi", "eltv",
}

DECIMALES = {
    "orig_upb", "orig_rate", "current_upb", "current_rate", "non_interest_upb",
    "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries", "total_expenses",
    "legal_costs", "maintenance_costs", "taxes_insurance", "misc_expenses",
    "actual_loss", "cum_mod_cost", "zero_balance_upb", "dlq_accrued_interest",
    "current_mod_cost", "interest_bearing_upb", "cramdown_costs",
}

# --------------------------------------------------------------------------
# Códigos de "no disponible" por columna (se convierten en nulos al leer)
# --------------------------------------------------------------------------
# Un vacío NO siempre es un dato faltante: en msa significa zona no
# metropolitana, en special_eligibility "no aplica", en zero_balance_code
# "crédito activo" y en mod_flag "nunca modificado". Esos vacíos se conservan
# como están y se interpretan en el análisis.

NO_DISPONIBLE = {
    "credit_score": ["9999"],
    "vantage_score": ["9999"],
    "first_time_homebuyer": ["9"],
    "mi_pct": ["999"],
    "num_units": ["99"],
    "occupancy": ["9"],
    "ocltv": ["999"],
    "dti": ["999"],
    "oltv": ["999"],
    "channel": ["9"],
    "prop_type": ["99"],
    "zip3": ["000"],
    "purpose": ["9"],
    "num_borrowers": ["99"],
    "valuation_method": ["7", "9"],
    "eltv": ["999"],
    "mi_cancel": ["7", "9"],
}

# Columnas de desempeño que solo se conocen después de que el crédito sale del
# portafolio. Sirven para describir pérdidas, pero nunca como variables de
# entrada de un modelo (fuga de información).
POSTERIORES_AL_CIERRE = [
    "zero_balance_date", "zero_balance_upb", "mi_recoveries", "net_sales_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_costs",
    "taxes_insurance", "misc_expenses", "actual_loss", "dlq_accrued_interest",
    "defect_settlement_date",
]

# Definición de incumplimiento del diccionario de datos
ESTADOS_INCUMPLIMIENTO_MIN = 3          # 90 días o más de atraso
CODIGOS_CIERRE_INCUMPLIMIENTO = (2, 3, 9)  # venta a tercero, venta corta o castigo, venta del REO
CODIGO_CIERRE_DEFECTO = 96               # defecto confirmado: se excluye, no es incumplimiento


# --------------------------------------------------------------------------
# Funciones de lectura
# --------------------------------------------------------------------------

def detectar_formato(ruta: str | Path, tipo: str) -> tuple[str, list[str]]:
    """Devuelve (formato, columnas) según el número de campos de la primera fila.

    tipo: "originacion" o "desempeno".
    """
    with open(ruta, "r", encoding="latin-1") as f:
        n_campos = f.readline().rstrip("\r\n").count("|") + 1
    try:
        return ESQUEMAS[(tipo, n_campos)]
    except KeyError as exc:
        raise ValueError(
            f"{ruta}: {n_campos} columnas; no corresponde a ningún formato conocido de {tipo}"
        ) from exc


def _expresion(col: str) -> str:
    """Expresión SQL que convierte una columna de texto crudo a su tipo final."""
    valor = f"trim({col})"
    codigos = NO_DISPONIBLE.get(col)
    if codigos:
        lista = ", ".join(f"'{c}'" for c in codigos)
        valor = f"CASE WHEN {valor} IN ({lista}) THEN NULL ELSE {valor} END"
    valor = f"nullif({valor}, '')"
    if col in ENTEROS:
        return f"TRY_CAST({valor} AS INTEGER) AS {col}"
    if col in DECIMALES:
        return f"TRY_CAST({valor} AS DOUBLE) AS {col}"
    return f"{valor} AS {col}"


def consulta_lectura(ruta: str | Path, tipo: str, columnas: list[str] | None = None) -> str:
    """Consulta DuckDB que lee un archivo crudo con nombres, tipos y nulos correctos.

    columnas: subconjunto a leer (por defecto todas). Las que no existan en el
    formato del archivo se devuelven como NULL, para que el resto del código no
    dependa del formato.
    """
    _, todas = detectar_formato(ruta, tipo)
    pedidas = columnas or todas
    spec = ", ".join(f"'{c}': 'VARCHAR'" for c in todas)
    seleccion = ", ".join(
        _expresion(c) if c in todas else f"NULL AS {c}" for c in pedidas
    )
    return (
        f"SELECT {seleccion} FROM read_csv('{Path(ruta).as_posix()}', delim='|', "
        f"header=false, quote='', escape='', columns={{{spec}}}, "
        f"null_padding=true, ignore_errors=false)"
    )


def consulta_cruda(ruta: str | Path, tipo: str) -> str:
    """Consulta que lee el archivo con todas sus columnas como texto, sin convertir nada.

    Sirve para medir códigos especiales y vacíos tal como vienen en el archivo.
    """
    _, todas = detectar_formato(ruta, tipo)
    spec = ", ".join(f"'{c}': 'VARCHAR'" for c in todas)
    return (
        f"SELECT * FROM read_csv('{Path(ruta).as_posix()}', delim='|', header=false, "
        f"quote='', escape='', columns={{{spec}}}, null_padding=true)"
    )
