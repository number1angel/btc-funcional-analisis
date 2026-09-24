# TP Final: Análisis y Pronóstico de BTC con Enfoque Funcional
**PROGRAMACIÓN FUNCIONAL · LICENCIATURA EN CIENCIA DE DATOS**  
**Cátedra:** Javier Epeloa · **Comisión:** 70AT  
**Alumnos:** Abraham Milena, Arguinzoniz Julieta, Juárez María Ailén, Muñoz Tadeo, Rossello Nicolás  
**Fecha de entrega:** Septiembre de 2026  

---

## 1. Introducción y Planteo del Problema

La pregunta central que orienta este trabajo es: *¿es posible pronosticar el precio de Bitcoin a un horizonte de una semana bajo principios matemáticos determinísticos y de programación funcional pura?*

Bitcoin (BTC) se caracteriza por una marcada volatilidad temporal y cambios bruscos de régimen. Cualquiera puede construir una curva que sobreajuste los datos del pasado; el verdadero desafío analítico radica en evaluar la capacidad predictiva sobre observaciones futuras no vistas durante la calibración.

### Restricción metodológica de la consigna
Por requerimiento explícito de la cátedra de Programación Funcional, **se descartó por completo el uso de librerías de Machine Learning (como `scikit-learn`) y de modelos predictivos de caja negra (Gradient Boosting, redes neuronales o autoregresivos tradicionales)**. En su lugar, el pipeline predictivo se restringió estrictamente a:
1. Formulaciones matemáticas cerradas y determinísticas.
2. Estimadores computados mediante funciones puras, inmutabilidad y funciones de orden superior (`map`, `reduce`).
3. Herramientas estándar de Python 3.12 (NumPy, SciPy para optimización no lineal, Statsmodels para tests diagnósticos y Matplotlib para visualización).

Trabajamos con la serie `btc-timeseries.json`, compuesta por 365 observaciones diarias consecutivas de BTC (donde cada precio corresponde al promedio diario $(\text{máx}+\text{mín})/2$). Con estos datos formulamos dos modelos determinísticos y los comparamos contra un pronóstico base ingenuo:
* **Modelo Base Ingenuo ($W=1$):** Asume que el precio de los próximos 7 días será idéntico al último valor observado.
* **Ventana Móvil ($W=7$ días):** Closure funcional que calcula el promedio de la última semana mediante `reduce` y proyecta ese valor constante.
* **Función No Lineal ($\text{Tanh} + \text{Fourier}$):** Curva continua parametrizada por una tangente hiperbólica (tendencia) sumada a cuatro armónicos de Fourier (oscilaciones), calibrada por mínimos cuadrados no lineales (`scipy.optimize.curve_fit`).

---

## 2. Objetivos

### 2.1 Objetivo general
Desarrollar un pipeline funcional puro de análisis y pronóstico para la serie de tiempo de Bitcoin, comparando un modelo no lineal y un estimador de ventana móvil frente a un baseline ingenuo mediante validación temporal exhaustiva.

### 2.2 Objetivos específicos
1. **Diagnóstico estadístico:** Evaluar formalmente la estacionariedad del precio y de sus retornos logarítmicos mediante la prueba de Dickey-Fuller Aumentada (ADF), cuantificar la autocorrelación serial y calcular la volatilidad anualizada.
2. **Implementación bajo paradigma funcional:** Implementar los modelos utilizando configuración inmutable (`@dataclass(frozen=True)`), closures, acumulación Monoidal con `reduce` y el patrón **Functor** sobre el tipo `FitOutcome`, **verificando formalmente mediante `assert` las leyes de Functor (identidad y composición) y de Monoid (elemento neutro y asociatividad)**.
3. **Comparación rigurosa de modelos:** Evaluar el error de predicción en 30 ventanas temporales independientes de 7 días (*walk-forward validation*), midiendo MAE y MAPE, y contrastar la significancia estadística frente al baseline mediante el test pareado de Wilcoxon con **corrección de Bonferroni**.
4. **Evaluación de sobreajuste y extrapolación:** Medir la degradación del error del modelo no lineal al extrapolar sobre un horizonte de 30 días no observados en comparación con su ajuste dentro de muestra (*in-sample*).

---

## 3. Metodología

### 3.1 Datos y herramientas de software
* **Dataset:** 365 observaciones diarias consecutivas desde el 29 de agosto de 2025 al 28 de agosto de 2026, sin valores faltantes ni discontinuidades. Cabe notar que al registrar $(\text{máx}+\text{mín})/2$ en lugar del precio de cierre, la serie presenta un suavizado artificial intradiario que induce cierta autocorrelación positiva de corto plazo.
* **Entorno:** Python 3.12, NumPy 2.4, pandas 3.0 (exclusivamente para lectura tabular y alineación de fechas), SciPy 1.17 (`curve_fit` y `wilcoxon`), Statsmodels 0.15 (`adfuller` y `acf`), Matplotlib 3.10, junto con `functools`, `dataclasses` y `typing`.
* **Métricas de error:** MAE, MAPE y RMSE fueron implementadas a mano como funciones puras sin dependencias externas.

### 3.2 Implementación funcional del estimador de Ventana Móvil
Para garantizar pureza y ausencia de efectos colaterales, el estimador de ventana móvil se estructuró mediante una fábrica de funciones de orden superior y reducción sobre tuplas inmutables:

```python
from functools import reduce
from typing import Callable, Tuple

def crear_estimador_ventana(w: int) -> Callable[[Tuple[float, ...], int], Tuple[float, ...]]:
    """Closure funcional puro: calcula la media de los últimos W días con reduce."""
    def estimador(historial: Tuple[float, ...], horizonte: int = 7) -> Tuple[float, ...]:
        ventana = historial[-w:]
        media = reduce(lambda a, b: a + b, ventana) / float(w)
        return (media,) * horizonte
    return estimador

estimador_w7 = crear_estimador_ventana(w=7)
```

### 3.3 Modelo No Lineal ($\text{Tanh} + \text{Fourier}$)
La función objetivo parametriza simultáneamente tendencia no lineal y periodicidad:
$$f(u) = a_0 + a_1 \tanh\big(a_2 (u - a_3)\big) + \sum_{k=1}^{K} \big[b_k \sin(\omega_k u) + c_k \cos(\omega_k u)\big]$$

Donde:
* $u = t / n \in [0, 1]$ es el tiempo normalizado.
* $K = 4$ armónicos trigonométricos (12 parámetros libres en total).
* $P \in \{2.0, 1.0, 0.5, 0.333, 0.25\}$ es el período fundamental evaluado.
* $\omega_k = \frac{2\pi k}{P}$ representa la frecuencia angular discreta de cada armónico.

Para manejar posibles fallos de convergencia de `curve_fit`, se diseñó el registro inmutable `FitOutcome`, el cual implementa el método `.map(f)` preservando la forma (Functor).

### 3.4 Decisiones de diseño metodológico
| Decisión | Elección | Justificación metodológica |
| :--- | :--- | :--- |
| **Baseline de contraste** | Repetir último precio ($W=1$) | En series con raíz unitaria / paseo aleatorio, el último valor es el benchmark canónico que todo modelo más complejo debe superar. |
| **Tamaño de ventana ($W$)** | 7 días | Corresponde al ciclo semanal completo de cotización continua de criptoactivos (24/7). |
| **Horizonte y avance ($h$)** | 7 días cada uno | Garantiza que las ventanas de prueba temporales sean disjuntas y ningún día compute dos veces en el error. |
| **Base inicial de entrenamiento** | 150 días | Provee suficientes grados de libertad para ajustar de forma estable los 12 parámetros de la curva no lineal y permite 30 ventanas de test. |
| **Métricas de evaluación** | MAE y MAPE | El MAE provee interpretación directa en USD; el MAPE permite comparar períodos de precios dispares (de USD 60.000 a USD 120.000). |
| **Test de hipótesis pareado** | Wilcoxon signed-rank | La distribución de errores entre ventanas no es normal (mediana 2,54 % vs. media 3,44 %, con colas asimétricas de hasta 14 %), invalidando el test t de Student. |
| **Ajuste por comparaciones múltiples** | Corrección de Bonferroni | Al contrastar dos modelos simultáneamente contra el base, el umbral de significancia se ajusta a $\alpha = 0,05 / 2 = 0,025$. |
| **Estructuras de datos** | Tuplas inmutables y reduce | Se adopta el patrón Monoid (neutro `()`, concatenación asociativa `+`) para acumular los 30 resultados sin mutar listas con `.append()`. |

---

## 4. Resultados

### 4.1 Resumen del pipeline de datos
* Observaciones totales: 365 días (29/08/2025 al 28/08/2026).
* Valores faltantes o descartados: 0.
* Retornos logarítmicos calculados: 364.
* Ventanas de validación evaluadas con éxito: 30 de 30 (100 % de convergencia).
* Leyes de Functor y Monoid verificadas formalmente: **4 de 4 con `assert`** (identidad y composición sobre `FitOutcome`; elemento neutro y asociatividad sobre tuplas).

### 4.2 Diagnóstico de la serie temporal
| Indicador | Valor numérico | Interpretación formal |
| :--- | :--- | :--- |
| **ADF sobre el precio** | Estadístico: -1,704 ($p = 0,4292$) | No se rechaza la hipótesis nula de raíz unitaria; el precio no es estacionario. |
| **ADF sobre log-retornos** | Estadístico: -6,423 ($p < 0,0001$) | Se rechaza raíz unitaria; los retornos diarios son estacionarios. |
| **Volatilidad diaria ($\sigma$)** | 1,84 % | Desvío estándar de los retornos logarítmicos diarios. |
| **Volatilidad anualizada** | 35,22 % | Estimada considerando 365 días de cotización ininterrumpida. |
| **Autocorrelación lag-1** | 0,35 | Correlación positiva moderada, en parte inducida por el suavizado $(\text{máx}+\text{mín})/2$. |

### 4.3 Comparación de desempeño predictivo (30 ventanas de 7 días)
| Modelo | MAE promedio (USD) | MAPE promedio (%) | MAPE mediano (%) | Ventanas ganadas frente al Base | Wilcoxon vs. Base ($p$-valor) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modelo Base ($W=1$)** | **2.392,70** | **3,44 %** | **2,54 %** | — | — |
| **Ventana Móvil ($W=7$)** | 2.899,27 | 4,16 % | 2,84 % | 12 de 30 (Base ganó 18) | $p = 0,0473$ |
| **Función No Lineal ($\text{Tanh}+\text{Fourier}$)** | 7.038,21 | 10,35 % | 8,62 % | 5 de 30 (Base ganó 25) | $p < 0,0001$ ($7,99 \times 10^{-6}$) |

Bajo el umbral riguroso de Bonferroni ($\alpha = 0,025$), la función no lineal resulta significativamente inferior al modelo base ($p < 0,0001$). En cambio, la diferencia entre el Modelo Base y la Ventana Móvil ($p = 0,0473 > 0,025$) **no alcanza significancia estadística estricta**, constituyendo una evidencia empírica débil o moderada a favor de la inmediatez del modelo base.

### 4.4 Evaluación de extrapolación fuera de muestra (Función No Lineal)
* **Ajuste in-sample (335 días de calibración):** $\text{MAE} = \text{USD } 2.501$, $\text{RMSE} = \text{USD } 3.276$, $\text{MAPE} = 3,11 \%$.
* **Extrapolación out-of-sample (30 días no vistos):** $\text{MAE} = \text{USD } 44.893$, $\text{RMSE} = \text{USD } 50.898$, $\text{MAPE} = 64,02 \%$.

El error fuera de muestra se multiplicó por un factor de 20, evidenciando un severo sobreajuste de los armónicos trigonométricos a la estructura histórica local.

### 4.5 Proyecciones a 30 días futuros
| Modelo | Día 1 proyectado (USD) | Día 30 proyectado (USD) | Comportamiento frente a la banda de volatilidad |
| :--- | :---: | :---: | :--- |
| **Modelo Base ($W=1$)** | 79.176,55 | 79.176,55 | Permanece en el centro de la banda por definición constructiva. |
| **Ventana Móvil ($W=7$)** | 78.515,01 | 78.515,01 | Se mantiene contenida dentro del cono de volatilidad ($\pm 2\sigma$). |
| **Función No Lineal** | 86.249,93 | 352.181,91 | Se descontrola exponencialmente, perforando la banda superior desde el día 1. |

### 4.6 Registro de incidencias y lecciones aprendidas
1. **Descarte de Machine Learning:** La versión inicial contemplaba `GradientBoostingRegressor`, la cual debió ser completamente reescrita bajo modelos funcionales determinísticos para cumplir con la consigna.
2. **Depuración de excepciones en optimización:** Inicialmente, capturar `Exception` de forma genérica dentro de `fit_lambda_function` enmascaró errores de nombres no definidos (`NameError`), reportándolos erróneamente como fallas numéricas de convergencia. Se corrigió acotando estrictamente el bloque `try/except` a errores propios del optimizador (`RuntimeError`, `ValueError`).

---

## 5. Conclusiones

### Cumplimiento de objetivos
* **Objetivo 1 (Diagnóstico): Cumplido.** Se constató la naturaleza no estacionaria del precio frente a la estacionariedad de los retornos, justificando el uso del benchmark ingenuo.
* **Objetivo 2 (Implementación funcional): Cumplido.** Se articularon dataclasses congeladas, closures puros con `reduce` y el tipo `FitOutcome`. Las leyes de Functor (identidad y composición) y las propiedades algebraicas de Monoid quedaron verificadas formalmente en el código mediante aserciones estrictas (`assert`).
* **Objetivo 3 (Comparación estadística): Cumplido.** El baseline superó a la función no lineal con contundencia estadística ($p < 0,0001$). Frente a la ventana móvil, si bien el base presentó menor error (3,44 % vs. 4,16 %), la prueba de Wilcoxon con corrección de Bonferroni ($p = 0,0473 > 0,025$) demostró que la ventaja no es categórica.
* **Objetivo 4 (Extrapolación y sobreajuste): Cumplido.** Se demostró que un excelente ajuste dentro de muestra (3,11 %) no garantiza validez predictiva externa, alcanzando un error del 64,02 % en extrapolación.

### Reflexión sobre el paradigma funcional en Python
El enfoque funcional resultó sumamente natural y elegante para estructurar el núcleo del modelado: la creación de closures configurables (`crear_estimador_ventana`), la evaluación sin efectos colaterales mediante `map` y la agregación monoidal con `reduce` eliminaron por completo el estado mutable intermedio.

Por el contrario, la fricción apareció en las interfaces externas: bibliotecas numéricas como NumPy y pandas operan internamente bajo paradigmas imperativos y vectorizados; asimismo, `curve_fit` comunica anomalías mediante excepciones en lugar de tipos algebraicos de error, obligándonos a encapsular manualmente los resultados en estructuras tipo `FitOutcome`.

### Limitaciones y trabajo futuro
* El análisis se restringió a un único activo y una sola ventana temporal anual ($N=365$).
* El precio diario representa $(\text{máx}+\text{mín})/2$ y no el precio de cierre oficial.
* Como trabajo futuro se propone evaluar ventanas alternativas ($W \in \{3, 14, 30\}$), extender `FitOutcome` a una Mónada completa incorporando `flat_map` (Either/Result), e implementar un estimador funcional de Suavizado Exponencial Simple (SES).

---

## 6. Declaración de Uso de Inteligencia Artificial

En cumplimiento con las normas de honestidad académica de la cátedra:
* Se utilizaron herramientas de modelos de lenguaje basados en IA (asistente de código y análisis) como soporte para:
  1. Revisión y contraste del código frente a las pautas de estilo funcional de la cátedra.
  2. Detección y corrección de excepciones enmascaradas en el bloque de optimización de `curve_fit`.
  3. Redacción y estructuración formal del informe técnico y verificación de inconsistencias entre salidas numéricas y redacción.
* Todo el código resultante fue verificado, ejecutado y validado de punta a punta por el equipo de alumnos sobre el entorno de ejecución local, garantizando la total comprensión y responsabilidad sobre los resultados presentados.

---

## 7. Referencias
* Apuntes de Cátedra de Programación Funcional, Licenciatura en Ciencia de Datos, Clases 1 a 4.
* McKinney, W. *Python for Data Analysis*, 3rd Edition, O'Reilly Media.
* SciPy Documentation: `scipy.optimize.curve_fit` & `scipy.stats.wilcoxon`.
