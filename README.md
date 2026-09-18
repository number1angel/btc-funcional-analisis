# BTC Functional Analysis

Trabajo de Programación Funcional aplicado al análisis y pronóstico de una serie temporal de Bitcoin.

El notebook compara dos enfoques que no utilizan una recta ni un polinomio:

- una función no lineal construida con tangente hiperbólica y armónicos de Fourier;
- un modelo de *machine learning* basado en Gradient Boosting y variables de rezago.

Los modelos se evalúan mediante *walk-forward validation* y se comparan con un modelo base ingenuo (*naive*).

## Contenido

```text
BTC_Analysis.ipynb    Notebook principal
btc-timeseries.json  Serie temporal utilizada por el análisis
requirements.txt     Dependencias de Python
CONTRIBUTING.md      Flujo de colaboración con branches
```

## Ejecución

El notebook usa `btc-timeseries.json` desde la misma carpeta cuando está disponible. Si se abre directamente en Google Colab, descarga automáticamente el dataset desde este repositorio.

### Google Colab

1. Abrir el notebook desde GitHub mediante **Archivo → Abrir cuaderno → GitHub**.
2. Seleccionar la branch correspondiente.
3. Ejecutar las celdas en orden con **Entorno de ejecución → Ejecutar todas**.

### Entorno local

```bash
git clone https://github.com/number1angel/btc-funcional-analisis.git
cd btc-funcional-analisis
python -m pip install -r requirements.txt
jupyter notebook BTC_Analysis.ipynb
```

## Metodología

El trabajo incorpora funciones puras, inmutabilidad, funciones de orden superior, closures, Functor, Monoid y una función de transición pura. El análisis incluye exploración de datos, diagnóstico de estacionariedad, autocorrelación, ajuste no lineal, Gradient Boosting, validación temporal y extrapolación a 30 días.

## Alcance

Este proyecto tiene fines académicos. Sus resultados no constituyen asesoramiento financiero ni deben utilizarse como única base para tomar decisiones de inversión.
