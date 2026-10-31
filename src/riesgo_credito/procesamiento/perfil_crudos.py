"""Perfil de los archivos crudos de Freddie Mac, sin descomprimirlos a disco.

Para cada trimestre cuenta filas, detecta el formato y, por columna, mide
vacíos, códigos de no disponible, mínimos y máximos, y los valores de las
columnas categóricas. Sirve para decidir con todos los trimestres qué
columnas se eliminan y qué valores debe manejar el ETL.

Uso, desde la raíz del repositorio y con las credenciales de AWS vigentes:
    PYTHONPATH=src python -m riesgo_credito.procesamiento.perfil_crudos --anios 2019 2020

Descarga cada zip anual del bucket, lo procesa trimestre por trimestre y lo
borra. Los trimestres ya medidos se saltan, así que se puede volver a ejecutar
si se interrumpe.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv

from riesgo_credito.procesamiento import esquema

BUCKET = "s3://riesgodecreditohipotecario"
MAX_VALORES = 150  # una columna con más valores distintos se deja de contar

CATEGORICAS = {
    "first_time_homebuyer", "occupancy", "channel", "ppm_flag", "amort_type", "state",
    "prop_type", "purpose", "super_conforming", "special_eligibility", "harp",
    "valuation_method", "io_flag", "num_units", "num_borrowers", "mi_cancel",
    "dlq_status", "mod_flag", "zero_balance_code", "step_mod_flag", "payment_deferral",
    "dlq_disaster", "assistance_plan",
}
NUMERICAS = (esquema.ENTEROS | esquema.DECIMALES) - CATEGORICAS


class Perfil:
    def __init__(self, columnas: list[str]):
        self.columnas = columnas
        self.filas = 0
        self.vacios = Counter()
        self.no_numericos = Counter()
        self.ceros = Counter()
        self.minimo, self.maximo = {}, {}
        self.valores = {c: Counter() for c in columnas if c in CATEGORICAS}
        self.codigos = {c: Counter() for c in columnas if c in esquema.NO_DISPONIBLE}

    def agregar(self, lote: pa.RecordBatch) -> None:
        self.filas += lote.num_rows
        for c in self.columnas:
            a = lote.column(c)
            self.vacios[c] += a.null_count
            if c in self.codigos:
                for codigo in esquema.NO_DISPONIBLE[c]:
                    self.codigos[c][codigo] += pc.sum(pc.equal(a, codigo)).as_py() or 0
            if c in self.valores and len(self.valores[c]) <= MAX_VALORES:
                for fila in pc.value_counts(a.drop_null()).to_pylist():
                    self.valores[c][fila["values"]] += fila["counts"]
            elif c in NUMERICAS:
                validos = a.drop_null()
                es_numero = pc.match_substring_regex(validos, r"^-?[0-9]+(\.[0-9]+)?$")
                self.no_numericos[c] += len(validos) - (pc.sum(es_numero).as_py() or 0)
                numeros = pc.cast(pc.filter(validos, es_numero), pa.float64())
                if len(numeros):
                    mm = pc.min_max(numeros).as_py()
                    self.minimo[c] = min(self.minimo.get(c, mm["min"]), mm["min"])
                    self.maximo[c] = max(self.maximo.get(c, mm["max"]), mm["max"])
                    self.ceros[c] += pc.sum(pc.equal(numeros, 0)).as_py() or 0

    def resultado(self) -> dict:
        salida = {}
        for c in self.columnas:
            d = {"vacios": self.vacios[c]}
            if c in self.codigos:
                d["no_disponible"] = dict(self.codigos[c])
            if c in self.valores:
                v = self.valores[c]
                d["valores"] = dict(v.most_common()) if len(v) <= MAX_VALORES else f"más de {MAX_VALORES}"
            if c in NUMERICAS:
                d.update(no_numericos=self.no_numericos[c], minimo=self.minimo.get(c),
                         maximo=self.maximo.get(c), ceros=self.ceros[c])
            salida[c] = d
        return salida


def perfilar_txt(z: zipfile.ZipFile, nombre: str, tipo: str) -> dict:
    with z.open(nombre) as f:
        n_campos = io.TextIOWrapper(f, encoding="latin-1").readline().rstrip("\r\n").count("|") + 1
    formato, columnas = esquema.ESQUEMAS[(tipo, n_campos)]
    perfil = Perfil(columnas)
    invalidas = []

    def fila_invalida(fila):
        invalidas.append(fila.number)
        return "skip"

    with z.open(nombre) as f:
        lector = csv.open_csv(
            f,
            read_options=csv.ReadOptions(column_names=columnas, encoding="latin1", block_size=1 << 26),
            parse_options=csv.ParseOptions(delimiter="|", quote_char=False, invalid_row_handler=fila_invalida),
            convert_options=csv.ConvertOptions(column_types={c: pa.string() for c in columnas},
                                               null_values=[""], strings_can_be_null=True),
        )
        for lote in lector:
            perfil.agregar(lote)
    return {"archivo": nombre, "formato": formato, "columnas": n_campos, "filas": perfil.filas,
            "filas_invalidas": len(invalidas), "bytes_descomprimido": z.getinfo(nombre).file_size,
            "perfil": perfil.resultado()}


def perfilar_trimestre(ruta_zip: Path) -> dict:
    with zipfile.ZipFile(ruta_zip) as z:
        nombres = [n for n in z.namelist() if n.endswith(".txt")]
        orig = [n for n in nombres if Path(n).name.startswith("orig_")
                or (Path(n).name.startswith("historical_data_") and "_time_" not in n)]
        perf = [n for n in nombres if Path(n).name.startswith("perf_") or "_time_" in n]
        return {"bytes_zip": ruta_zip.stat().st_size,
                "originacion": perfilar_txt(z, orig[0], "originacion"),
                "desempeno": perfilar_txt(z, perf[0], "desempeno")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Perfil de los crudos de Freddie Mac")
    parser.add_argument("--anios", nargs="+", default=[str(a) for a in range(2019, 2027)])
    parser.add_argument("--salida", default="data/metricas/perfil_crudos")
    parser.add_argument("--zip-local", help="perfilar un zip anual ya descargado, sin usar S3")
    args = parser.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    for anio in args.anios:
        temporal = Path(tempfile.mkdtemp(prefix=f"perfil_{anio}_", dir=Path.home()))
        try:
            if args.zip_local:
                anual = Path(args.zip_local)
            else:
                anual = temporal / f"historical_data_{anio}.zip"
                print(f"{anio}: descargando de S3...", flush=True)
                subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{anual.name}", str(anual), "--only-show-errors"],
                               check=True)
            with zipfile.ZipFile(anual) as z:
                trimestrales = sorted(n for n in z.namelist() if n.endswith(".zip"))
                if not trimestrales:
                    print(f"{anio}: el zip no trae zips trimestrales; contiene {z.namelist()[:10]}", flush=True)
                for nombre in trimestrales:
                    trimestre = Path(nombre).stem.replace("historical_data_", "")
                    destino = salida / f"{trimestre}.json"
                    if destino.exists():
                        print(f"{trimestre}: ya medido, se salta", flush=True)
                        continue
                    t0 = time.time()
                    ruta = Path(z.extract(nombre, temporal))
                    resultado = {"trimestre": trimestre, **perfilar_trimestre(ruta)}
                    resultado["segundos"] = round(time.time() - t0)
                    ruta.unlink()
                    destino.write_text(json.dumps(resultado, indent=1, ensure_ascii=False) + "\n")
                    print(f"{trimestre}: {resultado['originacion']['filas']:,} créditos, "
                          f"{resultado['desempeno']['filas']:,} registros, {resultado['segundos']} s", flush=True)
        finally:
            shutil.rmtree(temporal, ignore_errors=True)
    print("Terminado", flush=True)


if __name__ == "__main__":
    main()
