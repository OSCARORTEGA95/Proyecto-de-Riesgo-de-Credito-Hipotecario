"""KNN para las preguntas 1 y 2: conteo de datos, referencia del puntaje y versiones registradas en MLflow.

Uso, desde la raíz del repositorio:
    PYTHONPATH=src python -m riesgo_credito.modelos.knn --pregunta p1 --accion conteo
    PYTHONPATH=src python -m riesgo_credito.modelos.knn --pregunta p1 --accion version --version v1
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from riesgo_credito.modelos.knn_modelo import SIN_DATO, ModeloKNN, a_numero, preparar, probabilidad

RAIZ = Path(__file__).resolve().parents[3]
CONFIG = Path(os.environ.get("KNN_CONFIG", RAIZ / "config" / "modelos_knn.yaml"))
CACHE = RAIZ / "data" / "cache_knn"
EXPORTAR = RAIZ / "modelos" / "knn"

P1_NUM = ["credit_score", "dti", "oltv", "ocltv", "orig_rate", "orig_upb", "orig_term", "mi_pct",
          "num_borrowers", "num_units"]
P1_CAT = ["purpose", "occupancy", "channel", "first_time_homebuyer", "prop_type", "valuation_method"]
VARS_P1 = P1_NUM + P1_CAT + ["state"]
FORM_NUM = ["credit_score", "dti", "oltv", "orig_upb", "orig_rate", "orig_term"]
FORM_CAT = ["purpose", "prop_type", "state"]
P2_NUM = ["edad", "pct_saldo", "dif_tasa", "eltv", "atrasos_12m", "max_atraso_12m", "meses_desde_atraso",
          "indulgencia_12m"]
P2_BANDERAS = ["con_atraso_previo", "diferimiento", "diferimiento_mes", "saldo_sin_interes", "mora_desastre",
               "modificado"]
P2_CAT = ["plan_asistencia"]
VARS_P2 = P2_NUM + P2_BANDERAS + P2_CAT

# numéricas, banderas (0/1) y categóricas de cada conjunto
CONJUNTOS = {
    "base": (P1_NUM, [], P1_CAT),
    "base_estado": (P1_NUM, [], P1_CAT + ["state"]),
    "formulario": (FORM_NUM, [], FORM_CAT),
    "solo_mes": (P2_NUM, P2_BANDERAS, P2_CAT),
    "completa": (P2_NUM + P1_NUM, P2_BANDERAS, P2_CAT + P1_CAT),
}


def log(mensaje):
    print(f"[{time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


def cargar_config():
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset(ruta):
    return ds.dataset(RAIZ / ruta, format="parquet", partitioning="hive")


def inicio_trimestre(t):
    return int(t[:4]) * 100 + (int(t[-1]) - 1) * 3 + 1


def trimestres_hasta(ruta, hasta):
    nombres = [p.name.split("=", 1)[1] for p in (RAIZ / ruta).glob("trimestre=*")]
    return sorted(t for t in nombres if inicio_trimestre(t) <= hasta)


def en_cache(nombre, leer):
    CACHE.mkdir(parents=True, exist_ok=True)
    ruta, meta = CACHE / f"{nombre}.parquet", CACHE / f"{nombre}.json"
    if ruta.exists() and meta.exists():
        log(f"Leyendo de la caché: {ruta.name}")
        return pd.read_parquet(ruta), json.loads(meta.read_text())
    df, info = leer()
    df.to_parquet(ruta, index=False)
    meta.write_text(json.dumps(info))
    return df, info


# ---------- lectura de datos ----------

def leer_p1(cfg, trimestres):
    c = cfg["p1"]
    filtro = ds.field("trimestre").isin(trimestres) & ds.field(c["etiqueta"]).is_valid()
    df = dataset(c["ruta"]).to_table(columns=["loan_id"] + VARS_P1 + [c["etiqueta"]], filter=filtro).to_pandas()
    df[c["etiqueta"]] = df[c["etiqueta"]].astype(int)
    return df.sort_values("loan_id", kind="stable").reset_index(drop=True)


def filtro_p2(c, desde, hasta):
    return (ds.field("trimestre").isin(trimestres_hasta(c["ruta"], hasta))
            & (ds.field("period") >= desde) & (ds.field("period") <= hasta)
            & (ds.field("dlq_num") == 0) & ds.field(c["etiqueta"]).is_valid())


def unir_p1(c, df):
    ids = df["loan_id"].unique().tolist()
    p1 = dataset(c["ruta_p1"]).to_table(columns=["loan_id"] + VARS_P1,
                                        filter=ds.field("loan_id").isin(ids)).to_pandas()
    df = df.merge(p1, on="loan_id", how="left")
    df[c["etiqueta"]] = df[c["etiqueta"]].astype(int)
    return df.sort_values(["loan_id", "period"], kind="stable").reset_index(drop=True)


def p2_entrenamiento(cfg, desde, hasta):
    c, y = cfg["p2"], cfg["p2"]["etiqueta"]
    g = c["grupos_entrenamiento"]
    columnas = ["loan_id", "period", "grupo_muestra", y] + VARS_P2

    def leer():
        d, f = dataset(c["ruta"]), filtro_p2(c, desde, hasta)
        log(f"P2 {desde}-{hasta}: leyendo positivos")
        pos = d.to_table(columns=columnas, filter=f & (ds.field(y) == 1)).to_pandas()
        log(f"P2 {desde}-{hasta}: leyendo negativos con grupo_muestra < {g}")
        neg = d.to_table(columns=columnas, filter=f & (ds.field(y) == 0) & (ds.field("grupo_muestra") < g)).to_pandas()
        log(f"P2 {desde}-{hasta}: contando todos los negativos")
        total_neg = d.count_rows(filter=f & (ds.field(y) == 0))
        df = unir_p1(c, pd.concat([pos, neg], ignore_index=True))
        return df, {"total_positivos": len(pos), "total_negativos": int(total_neg)}

    return en_cache(f"p2_entrenamiento_{desde}_{hasta}_g{g}", leer)


def p2_evaluacion(cfg, desde, hasta, g):
    c = cfg["p2"]
    columnas = ["loan_id", "period", "grupo_muestra", c["etiqueta"]] + VARS_P2

    def leer():
        log(f"P2 {desde}-{hasta}: leyendo la muestra con grupo_muestra >= {100 - g}")
        f = filtro_p2(c, desde, hasta) & (ds.field("grupo_muestra") >= 100 - g)
        return unir_p1(c, dataset(c["ruta"]).to_table(columns=columnas, filter=f).to_pandas()), {}

    return en_cache(f"p2_evaluacion_{desde}_{hasta}_g{g}", leer)


def datos(cfg, pregunta, etapa):
    c = cfg[pregunta]
    if pregunta == "p1":
        df = leer_p1(cfg, c[etapa])
        if etapa == "validacion" and len(df) > c["max_validacion"]:
            df = df.sample(n=c["max_validacion"], random_state=cfg["semilla"]).sort_index()
        return df, {}
    desde, hasta = c[etapa]
    if etapa in ("seleccion", "entrenamiento"):
        return p2_entrenamiento(cfg, desde, hasta)
    return p2_evaluacion(cfg, desde, hasta, c[f"grupos_{etapa}"])


def descripcion_etapa(cfg, pregunta, etapa):
    v = cfg[pregunta][etapa]
    return f"{v[0]}-{v[-1]}"


def regla_evaluacion(cfg, pregunta, etapa):
    if pregunta == "p1":
        return "todos los créditos" if etapa == "prueba" else f"muestra aleatoria de hasta {cfg['p1']['max_validacion']}"
    return f"grupo_muestra >= {100 - cfg['p2'][f'grupos_{etapa}']}"


# ---------- muestreo, modelo y métricas ----------

def muestrear(df, y, proporcion, maximo, semilla, total_positivos=None, total_negativos=None):
    pos, neg = df[df[y] == 1], df[df[y] == 0]
    total_positivos = total_positivos or len(pos)
    total_negativos = total_negativos or len(neg)
    if maximo and len(pos) * (1 + proporcion) > maximo:
        pos = pos.sample(n=int(maximo // (1 + proporcion)), random_state=semilla)
    n_neg = min(len(neg), len(pos) * proporcion)
    if n_neg < len(pos) * proporcion:
        log(f"Aviso: solo hay {len(neg)} negativos para {len(pos)} positivos; la proporción queda en 1 a {n_neg / len(pos):.1f}")
    neg = neg.sample(n=n_neg, random_state=semilla)
    frac_pos, frac_neg = len(pos) / total_positivos, len(neg) / total_negativos
    info = {"positivos": len(pos), "negativos": len(neg), "frac_positivos": frac_pos,
            "frac_negativos": frac_neg, "r": frac_neg / frac_pos}
    return pd.concat([pos, neg]).sort_index(), info


def crear_preparacion(numericas, categoricas):
    pasos = [("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]),
              numericas)]
    if categoricas:
        pasos.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas))
    return Pipeline([("prep", ColumnTransformer(pasos)), ("float32", FunctionTransformer(np.float32))])


def crear_pipeline(numericas, categoricas, k, pesos):
    prep = crear_preparacion(numericas, categoricas)
    return Pipeline(prep.steps + [("knn", KNeighborsClassifier(n_neighbors=k, weights=pesos, n_jobs=-1))])


def probabilidades_por_k(A, ya, B, ks, pesos, lote):
    nn = NearestNeighbors(n_neighbors=max(ks), n_jobs=-1).fit(A)
    salida = {k: np.empty(len(B)) for k in ks}
    for i in range(0, len(B), lote):
        dist, idx = nn.kneighbors(B[i:i + lote])
        yv = ya[idx]
        for k in ks:
            if pesos == "uniform":
                p = yv[:, :k].mean(axis=1)
            else:
                d = dist[:, :k].astype("float64")
                with np.errstate(divide="ignore"):
                    w = 1.0 / d
                cero = np.isinf(w)
                fila = cero.any(axis=1)
                w[fila] = cero[fila]
                p = (w * yv[:, :k]).sum(axis=1) / w.sum(axis=1)
            salida[k][i:i + lote] = p
    return salida


def metricas(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n_top = math.ceil(0.1 * len(y))
    top = np.argsort(-p, kind="stable")[:n_top]
    captura = y[top].sum() / y.sum()
    return {"auc_roc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p),
            "captura_decil": captura, "precision_decil": y[top].mean(), "lift_decil": captura / (n_top / len(y)),
            "tasa_observada": y.mean(), "prob_media": p.mean(), "n_prueba": len(y), "positivos_prueba": int(y.sum())}


# ---------- registro y exportación ----------

def commit_actual():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True)
    return r.stdout.strip() or "sin_git"


def requisitos():
    return [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}", f"numpy=={np.__version__}",
            f"joblib=={joblib.__version__}", f"mlflow=={mlflow.__version__}"]


def copiar_codigo(destino):
    base = destino / "riesgo_credito"
    (base / "modelos").mkdir(parents=True)
    origen = RAIZ / "src" / "riesgo_credito"
    for archivo in ["__init__.py", "modelos/__init__.py", "modelos/knn_modelo.py"]:
        shutil.copy(origen / archivo, base / archivo)
    return str(base)


def preparar_mlflow(cfg, pregunta):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit("Falta MLFLOW_TRACKING_URI (por ejemplo: export MLFLOW_TRACKING_URI=http://localhost:8050)")
    mlflow.set_experiment(cfg["mlflow"][pregunta])


def entrada_json(df, numericas, banderas, categoricas):
    """Tipos simples (decimal, texto o booleano) para la firma y el ejemplo del modelo en MLflow."""
    e = df.reset_index(drop=True).copy()
    for c in numericas:
        e[c] = a_numero(e[c])
    for c in banderas + categoricas:
        if pd.api.types.is_numeric_dtype(e[c]) and not pd.api.types.is_bool_dtype(e[c]):
            e[c] = a_numero(e[c])
        elif not (pd.api.types.is_bool_dtype(e[c]) and not e[c].isna().any()):
            e[c] = e[c].astype(object).where(e[c].notna(), None)
            if e[c].isna().all():
                e.loc[0, c] = SIN_DATO if c in categoricas else False
    return e


def guardar_json(ruta, datos_json):
    ruta.write_text(json.dumps(datos_json, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def actualizar_resumen():
    filas = []
    for metr in sorted(EXPORTAR.glob("p*/*/metricas.json")):
        fila = json.loads((metr.parent / "parametros.json").read_text(encoding="utf-8"))
        fila.update(json.loads(metr.read_text(encoding="utf-8")))
        filas.append(fila)
    if filas:
        pd.DataFrame(filas).to_csv(EXPORTAR / "resumen.csv", index=False)


def registrar(cfg, pregunta, nombre, params, metr, modelo=None, pipeline=None, ejemplo=None):
    run_name = f"{pregunta}_{nombre}"
    destino = EXPORTAR / pregunta / nombre.split("_")[0]
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        argumentos = {}
        if modelo is not None:
            ruta_pipe = tmp / "pipeline.joblib"
            joblib.dump(pipeline, ruta_pipe, compress=3)
            metr["tamano_modelo_mb"] = ruta_pipe.stat().st_size / 1e6
            modelo.predict(None, ejemplo)  # verifica que el ejemplo se pueda predecir
            # sin firma: MLflow rechazaría enteros en columnas decimales, y preparar() ya normaliza los tipos
            argumentos = dict(python_model=modelo, artifacts={"pipeline": str(ruta_pipe)},
                              code_paths=[copiar_codigo(tmp / "codigo")], signature=False,
                              input_example=ejemplo.head(5), pip_requirements=requisitos())
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({"git_commit": commit_actual(), "pregunta": pregunta, "version": nombre.split("_")[0]})
            mlflow.log_params(params)
            mlflow.log_metrics(metr)
            if argumentos:
                mlflow.pyfunc.log_model(name="modelo", **argumentos)
            run_id = run.info.run_id
        if argumentos:
            mlflow.pyfunc.save_model(path=str(destino / "modelo"), **argumentos)
    params = {"experimento": cfg["mlflow"][pregunta], "run_name": run_name, "run_id": run_id, **params}
    guardar_json(destino / "parametros.json", params)
    guardar_json(destino / "metricas.json", metr)
    actualizar_resumen()
    log(f"Registrado {run_name} (run_id {run_id}) y exportado en {destino.relative_to(RAIZ)}")


# ---------- acciones ----------

def conteo(cfg, pregunta):
    c, y = cfg[pregunta], cfg[pregunta]["etiqueta"]
    print(dataset(c["ruta"]).schema.to_string(show_schema_metadata=False), flush=True)
    if pregunta == "p1":
        df = dataset(c["ruta"]).to_table(columns=["trimestre", y]).to_pandas()
        tabla = df.groupby("trimestre").agg(creditos=(y, "size"), etiquetables=(y, "count"), positivos=(y, "sum"))
        tabla["tasa_pct"] = (100 * tabla["positivos"] / tabla["etiquetables"]).round(3)
        for etapa in ["seleccion", "validacion", "entrenamiento", "prueba"]:
            t = tabla.loc[c[etapa]]
            print(f"\n{etapa} ({descripcion_etapa(cfg, pregunta, etapa)})")
            print(t.to_string())
            print(f"Total: {t['etiquetables'].sum()} etiquetables, {int(t['positivos'].sum())} positivos, "
                  f"tasa {100 * t['positivos'].sum() / t['etiquetables'].sum():.3f} %")
        return
    desde = min(c["seleccion"][0], c["prueba"][0])
    hasta = max(c["entrenamiento"][1], c["prueba"][1])
    a0 = desde // 100
    n_meses = (hasta // 100 - a0) * 12 + hasta % 100
    conteos = np.zeros(n_meses * 2 * 100, dtype=np.int64)
    log(f"Recorriendo P2 entre {desde} y {hasta} (puede tardar varios minutos)")
    for lote in dataset(c["ruta"]).to_batches(columns=["period", y, "grupo_muestra"], filter=filtro_p2(c, desde, hasta)):
        per = lote.column("period").to_numpy(zero_copy_only=False).astype(np.int64)
        yy = lote.column(y).to_numpy(zero_copy_only=False).astype(np.int64)
        gg = lote.column("grupo_muestra").to_numpy(zero_copy_only=False).astype(np.int64)
        i = (per // 100 - a0) * 12 + per % 100 - 1
        conteos += np.bincount((i * 2 + yy) * 100 + gg, minlength=conteos.size)
    conteos = conteos.reshape(n_meses, 2, 100)
    periodos = [(a0 + m // 12) * 100 + m % 12 + 1 for m in range(n_meses)]
    mensual = pd.DataFrame({"period": periodos, "filas": conteos.sum(axis=(1, 2)), "positivos": conteos[:, 1, :].sum(axis=1)})
    mensual = mensual[mensual["period"] >= desde].set_index("period")
    mensual["tasa_pct"] = (100 * mensual["positivos"] / mensual["filas"]).round(4)
    print("\nPor mes")
    print(mensual.to_string())
    for etapa in ["seleccion", "validacion", "entrenamiento", "prueba"]:
        d0, d1 = c[etapa]
        sel = [periodos.index(p) for p in periodos if d0 <= p <= d1]
        bloque = conteos[sel]
        filas, pos = bloque.sum(), bloque[:, 1, :].sum()
        print(f"\n{etapa} ({d0}-{d1}): {filas} filas, {pos} positivos, tasa {100 * pos / filas:.4f} %")
        if etapa in ("seleccion", "entrenamiento"):
            for g in [1, 2, 5, 10, 20, 50]:
                print(f"  negativos con grupo_muestra < {g}: {bloque[:, 0, :g].sum()}")
        else:
            for g in [1, 2, 5, 10]:
                b = bloque[:, :, 100 - g:]
                print(f"  muestra grupo_muestra >= {100 - g}: {b.sum()} filas, {b[:, 1, :].sum()} positivos")


def tiempo(cfg, pregunta):
    v = cfg["versiones"][pregunta]["v1"]
    numericas, banderas, categoricas = CONJUNTOS[v["variables"]]
    y, semilla = cfg[pregunta]["etiqueta"], cfg["semilla"]
    ent, info = datos(cfg, pregunta, "entrenamiento")
    m, muestra = muestrear(ent, y, v["proporcion"], cfg[pregunta]["max_entrenamiento"], semilla, **info)
    prep = crear_preparacion(numericas + banderas, categoricas)
    A = prep.fit_transform(preparar(m, numericas, banderas, categoricas))
    B = A[np.random.default_rng(semilla).choice(len(A), size=min(5000, len(A)), replace=False)]
    nn = NearestNeighbors(n_neighbors=max(cfg["k_candidatos"]), n_jobs=-1).fit(A)
    t0 = time.time()
    nn.kneighbors(B)
    por_fila = (time.time() - t0) / len(B)
    print(f"\nEntrenamiento de v1: {len(m)} filas ({muestra['positivos']} positivos), {A.shape[1]} columnas codificadas")
    print(f"Tiempo por fila predicha: {1000 * por_fila:.2f} ms")
    for etapa in ["validacion", "prueba"]:
        df, _ = datos(cfg, pregunta, etapa)
        print(f"{etapa}: {len(df)} filas -> unos {len(df) * por_fila / 60:.1f} minutos de predicción por versión")


def correr_referencia(cfg, pregunta):
    preparar_mlflow(cfg, pregunta)
    y = cfg[pregunta]["etiqueta"]
    prueba, _ = datos(cfg, pregunta, "prueba")
    puntaje = a_numero(prueba["credit_score"])
    metr = metricas(prueba[y], -puntaje.fillna(puntaje.median()))
    metr.pop("prob_media")
    params = {"pregunta": pregunta, "modelo": "puntaje crediticio solo (menor puntaje, mayor riesgo)",
              "periodo_prueba": descripcion_etapa(cfg, pregunta, "prueba"),
              "regla_prueba": regla_evaluacion(cfg, pregunta, "prueba")}
    log(f"Referencia en prueba: AUC-ROC {metr['auc_roc']:.4f}, AUC-PR {metr['auc_pr']:.4f}, "
        f"captura del decil superior {100 * metr['captura_decil']:.1f} %")
    registrar(cfg, pregunta, "referencia_puntaje", params, metr)


def correr_version(cfg, pregunta, version):
    preparar_mlflow(cfg, pregunta)
    v = cfg["versiones"][pregunta][version]
    numericas, banderas, categoricas = CONJUNTOS[v["variables"]]
    y, semilla, ks, lote = cfg[pregunta]["etiqueta"], cfg["semilla"], cfg["k_candidatos"], cfg["lote_prediccion"]
    maximo = cfg[pregunta]["max_entrenamiento"]
    log(f"{pregunta} {version} ({v['descripcion']}): selección de k")
    t0 = time.time()
    sel, info = datos(cfg, pregunta, "seleccion")
    sel, _ = muestrear(sel, y, v["proporcion"], maximo, semilla, **info)
    val, _ = datos(cfg, pregunta, "validacion")
    prep = crear_preparacion(numericas + banderas, categoricas)
    A = prep.fit_transform(preparar(sel, numericas, banderas, categoricas))
    B = prep.transform(preparar(val, numericas, banderas, categoricas))
    probs = probabilidades_por_k(A, sel[y].to_numpy(), B, ks, v["pesos"], lote)
    aucs = {k: roc_auc_score(val[y], p) for k, p in probs.items()}
    k = max(aucs, key=aucs.get)
    t_sel = time.time() - t0
    log("AUC en validación por k: " + ", ".join(f"k={kk}: {a:.4f}" for kk, a in aucs.items()) + f" -> k = {k}")
    del sel, val, A, B, probs

    log("Entrenamiento final")
    ent, info = datos(cfg, pregunta, "entrenamiento")
    ent, muestra = muestrear(ent, y, v["proporcion"], maximo, semilla, **info)
    modelo = ModeloKNN(numericas, banderas, categoricas, muestra["r"])
    t1 = time.time()
    modelo.pipeline = crear_pipeline(numericas + banderas, categoricas, k, v["pesos"])
    modelo.pipeline.fit(preparar(ent, numericas, banderas, categoricas), ent[y].to_numpy())
    t_fit = time.time() - t1
    n_columnas = modelo.pipeline[:-1].transform(preparar(ent.head(1), numericas, banderas, categoricas)).shape[1]
    del ent

    log("Predicción sobre prueba")
    prueba, _ = datos(cfg, pregunta, "prueba")
    t2 = time.time()
    p = probabilidad(modelo.pipeline, preparar(prueba, numericas, banderas, categoricas), muestra["r"], lote)
    t_pred = time.time() - t2
    metr = metricas(prueba[y], p)
    metr.update({f"auc_validacion_k{kk}": a for kk, a in aucs.items()})
    metr.update({"auc_validacion": aucs[k], "tiempo_seleccion_s": t_sel, "tiempo_entrenamiento_s": t_fit,
                 "tiempo_prediccion_s": t_pred, "n_entrenamiento": muestra["positivos"] + muestra["negativos"],
                 "positivos_entrenamiento": muestra["positivos"]})
    params = {"pregunta": pregunta, "version": version, "descripcion": v["descripcion"],
              "variables": v["variables"], "lista_variables": ",".join(numericas + banderas + categoricas),
              "n_columnas_codificadas": n_columnas, "k": k, "k_candidatos": ",".join(map(str, ks)),
              "pesos": v["pesos"], "proporcion_negativos": v["proporcion"], "r": muestra["r"],
              "frac_positivos": muestra["frac_positivos"], "frac_negativos": muestra["frac_negativos"],
              "periodo_seleccion": descripcion_etapa(cfg, pregunta, "seleccion"),
              "periodo_validacion": descripcion_etapa(cfg, pregunta, "validacion"),
              "periodo_entrenamiento": descripcion_etapa(cfg, pregunta, "entrenamiento"),
              "periodo_prueba": descripcion_etapa(cfg, pregunta, "prueba"),
              "regla_validacion": regla_evaluacion(cfg, pregunta, "validacion"),
              "regla_prueba": regla_evaluacion(cfg, pregunta, "prueba"), "semilla": semilla}
    log(f"Prueba: AUC-ROC {metr['auc_roc']:.4f}, AUC-PR {metr['auc_pr']:.4f}, captura del decil superior "
        f"{100 * metr['captura_decil']:.1f} %, probabilidad media {metr['prob_media']:.5f} frente a tasa "
        f"{metr['tasa_observada']:.5f}")
    columnas = (["loan_id", "period"] if pregunta == "p2" else ["loan_id"]) + numericas + banderas + categoricas
    muestra_firma = entrada_json(prueba[columnas].sample(n=min(2000, len(prueba)), random_state=semilla),
                                 numericas, banderas, categoricas)
    registrar(cfg, pregunta, f"{version}_{v['descripcion']}", params, metr, modelo, modelo.pipeline, muestra_firma)


def main():
    parser = argparse.ArgumentParser(description="Modelos KNN de las preguntas 1 y 2")
    parser.add_argument("--pregunta", choices=["p1", "p2"], required=True)
    parser.add_argument("--accion", choices=["conteo", "tiempo", "referencia", "version"], required=True)
    parser.add_argument("--version", nargs="+", default=[])
    args = parser.parse_args()
    cfg = cargar_config()
    inicio = time.time()
    if args.accion == "conteo":
        conteo(cfg, args.pregunta)
    elif args.accion == "tiempo":
        tiempo(cfg, args.pregunta)
    elif args.accion == "referencia":
        correr_referencia(cfg, args.pregunta)
    else:
        if not args.version:
            raise SystemExit("Indica la versión, por ejemplo --version v1")
        for version in args.version:
            correr_version(cfg, args.pregunta, version)
    log(f"Terminó en {(time.time() - inicio) / 60:.1f} minutos")


if __name__ == "__main__":
    main()
