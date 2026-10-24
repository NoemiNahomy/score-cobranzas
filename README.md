# 📊 Score de Cobranzas

Prueba teórica de un **score de cobranzas** que estima la probabilidad de que un cliente de tarjeta de crédito **en mora** pague el mes siguiente, y la traduce en un puntaje de 0 a 1000 con una acción de cobranza sugerida.

## 🚀 App en Streamlit

La aplicación (`streamlit_app.py`) permite:

- 👤 **Cliente individual:** ingresar los datos de un cliente en mora y obtener su score, segmento y acción sugerida.
- 📁 **Cartera:** subir un CSV con la cartera en mora (o usar el dataset de prueba), calificar a todos los clientes y descargar la lista priorizada.
- ⬇️ **Dataset de prueba:** ver y descargar el dataset en CSV o Excel (con diccionario de datos).
- ℹ️ **Acerca del modelo:** métricas y segmentos.

### Ejecutar localmente

```bash
git clone https://github.com/<tu-usuario>/score-cobranzas.git
cd score-cobranzas
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🗂️ Diccionario de datos

Convención de nombres: `COD_` código · `MTO_` monto en bolivianos (Bs) · `EST_` estado · `IND_` indicador · sufijo `M0` = mes de corte (agosto 2026), `M1…M5` = meses anteriores.

| Variable                | Descripción                                                                             | Tipo       |
| ----------------------- | --------------------------------------------------------------------------------------- | ---------- |
| `COD_CLIENTE`           | Código interno del cliente                                                              | Entero     |
| `MTO_LIMITE_CREDITO`    | Límite de crédito de la tarjeta, incluye adicionales                                    | Bs         |
| `COD_SEXO`              | 1 = hombre; 2 = mujer                                                                   | Categórica |
| `COD_NIVEL_EDUCATIVO`   | 1 = posgrado; 2 = universidad; 3 = secundaria/técnico; 4 = otros                        | Categórica |
| `COD_ESTADO_CIVIL`      | 1 = casado; 2 = soltero; 3 = otros                                                      | Categórica |
| `EDAD`                  | Edad del cliente                                                                        | Años       |
| `DEPARTAMENTO`          | Departamento de Bolivia donde se administra la cuenta                                   | Categórica |
| `TIPO_EMPLEO`           | Dependiente, Independiente, Comerciante, Jubilado                                       | Categórica |
| `EST_PAGO_M0 … M5`      | Estado de pago: −2 sin consumo; −1 pagó el total; 0 pagó el mínimo; 1…8 meses de atraso | Ordinal    |
| `MTO_FACTURADO_M0 … M5` | Monto facturado en el extracto (negativo = saldo a favor)                               | Bs         |
| `MTO_PAGADO_M0 … M5`    | Monto pagado en el mes                                                                  | Bs         |
| `IND_INCUMPLE_MES_SIG`  | **Objetivo:** 1 = no paga el mes siguiente; 0 = paga                                    | Binaria    |

---

## 🧠 Modelo

- **Universo:** clientes en mora al corte (`EST_PAGO_M0 ≥ 1`): 5.133 clientes; el 48,8 % paga.
- **Variables derivadas:** `MAX_ATRASO`, `N_MESES_ATRASO`, `TENDENCIA_ATRASO`, `UTILIZACION`, `RATIO_PAGO`, `PROM_FACTURADO`, `PROM_PAGADO`.
- **Algoritmos comparados (8):** regresión logística, árbol de decisión, Random Forest, Gradient Boosting, SVM, red neuronal MLP, KNN y Naive Bayes (validación cruzada de 5 particiones + GridSearchCV).
- **Modelo elegido:** regresión logística con regularización L1 (C = 0,1), **sin variables demográficas** (sexo, educación, estado civil, edad).
- **Desempeño en prueba (30 %):** AUC = 0,813 · KS = 0,48.

| Segmento                  | Score    | Acción sugerida                                              |
| ------------------------- | -------- | ------------------------------------------------------------ |
| Alta probabilidad de pago | 650–1000 | Recordatorio automático y facilidades de pago en línea       |
| Probabilidad media        | 400–649  | Llamada prioritaria, promesa o acuerdo de pago               |
| Baja probabilidad de pago | 0–399    | Gestión especializada, reprogramación o cobranza prejudicial |

---
