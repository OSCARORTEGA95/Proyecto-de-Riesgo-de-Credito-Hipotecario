# model-pkg

Aquí va el modelo empaquetado que instala la API (línea final de `../requirements.txt`).

Después de construirlo en `package-src` (`tox run -e train`, `tox run -e test_package`, `python3 -m build`):

```bash
cp ../package-src/dist/modelo_riesgo_hipotecario-0.0.1-py3-none-any.whl .
```

El `.whl` lleva dentro el modelo entrenado con los datos de Freddie Mac, que no se pueden
redistribuir. Por eso no se sube a GitHub (ver `.gitignore`): se construye en el EC2.
