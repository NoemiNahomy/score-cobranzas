"""Lógica compartida del Score de Cobranzas (app, script de entrenamiento y notebook).

Convención de nombres: COD_ = código, MTO_ = monto en Bs, EST_ = estado, IND_ = indicador;
sufijo M0 = mes de corte, M1…M5 = meses anteriores.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent
DATA_CSV = RAIZ / "data" / "dataset.csv"

MODELO_PKL = RAIZ / "models" / "modelo_score_cobranzas.pkl"
RS = 42

ESTADOS = [f"EST_PAGO_M{i}" for i in range(6)]
FACTURADOS = [f"MTO_FACTURADO_M{i}" for i in range(6)]
PAGADOS = [f"MTO_PAGADO_M{i}" for i in range(6)]
OBJETIVO = "IND_INCUMPLE_MES_SIG"
DEPARTAMENTOS = ["Santa Cruz", "La Paz", "Cochabamba", "Beni", "Tarija", "Chuquisaca", "Oruro", "Potosí", "Pando"]
EMPLEOS = ["Dependiente", "Independiente", "Comerciante", "Jubilado"]

DERIVADAS = ["MAX_ATRASO", "N_MESES_ATRASO", "TENDENCIA_ATRASO", "UTILIZACION", "RATIO_PAGO", "PROM_FACTURADO", "PROM_PAGADO"]
MONTOS = ["MTO_LIMITE_CREDITO"] + FACTURADOS + PAGADOS + ["PROM_FACTURADO", "PROM_PAGADO"]
# Modelo de producción: sin variables demográficas (sexo, educación, estado civil, edad)
NUMERICAS = ["MTO_LIMITE_CREDITO"] + ESTADOS + FACTURADOS + PAGADOS + DERIVADAS
CATEGORICAS = ["DEPARTAMENTO", "TIPO_EMPLEO"]
VARIABLES = NUMERICAS + CATEGORICAS

SEGMENTOS = [  # (nombre, score mínimo, acción sugerida)
    ("Alta probabilidad de pago", 650, "Recordatorio automático (SMS, WhatsApp, banca móvil) y facilidades de pago en línea"),
    ("Probabilidad media", 400, "Llamada prioritaria, promesa o acuerdo de pago con seguimiento"),
    ("Baja probabilidad de pago", 0, "Gestión especializada, reprogramación o cobranza prejudicial"),
]


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza y variables derivadas (idénticas al notebook)."""
    df = df.copy()
    if "COD_NIVEL_EDUCATIVO" in df: df["COD_NIVEL_EDUCATIVO"] = df.COD_NIVEL_EDUCATIVO.replace({0: 4, 5: 4, 6: 4})
    if "COD_ESTADO_CIVIL" in df: df["COD_ESTADO_CIVIL"] = df.COD_ESTADO_CIVIL.replace({0: 3})
    for c in ESTADOS: df[c] = df[c].clip(lower=0)  # −2, −1, 0 → 0 (sin atraso)
    df["MAX_ATRASO"] = df[ESTADOS].max(axis=1)
    df["N_MESES_ATRASO"] = (df[ESTADOS] > 0).sum(axis=1)
    df["TENDENCIA_ATRASO"] = df.EST_PAGO_M0 - df.EST_PAGO_M1
    df["UTILIZACION"] = (df.MTO_FACTURADO_M0 / df.MTO_LIMITE_CREDITO).clip(-1, 3)
    base = df.MTO_FACTURADO_M1.where(df.MTO_FACTURADO_M1 > 0, 1)
    df["RATIO_PAGO"] = np.where(df.MTO_FACTURADO_M1 > 0, df.MTO_PAGADO_M0 / base, 1.0).clip(0, 2)
    df["PROM_FACTURADO"] = df[FACTURADOS].mean(axis=1)
    df["PROM_PAGADO"] = df[PAGADOS].mean(axis=1)
    return df


def universo_en_mora(df: pd.DataFrame) -> pd.DataFrame:
    """Clientes en mora al corte, sin duplicados, con variables preparadas."""
    m = df[df.EST_PAGO_M0 >= 1]
    m = m[~m.drop(columns="COD_CLIENTE", errors="ignore").duplicated()]
    return preparar(m)


def construir_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("montos", Pipeline([("arcsinh", FunctionTransformer(np.arcsinh, feature_names_out="one-to-one")),
                             ("esc", StandardScaler())]), [c for c in NUMERICAS if c in MONTOS]),
        ("num", StandardScaler(), [c for c in NUMERICAS if c not in MONTOS]),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CATEGORICAS),
    ])
    # Regularización L1 (C = 0,1). Desde scikit-learn 1.8 se expresa con l1_ratio=1; antes con penalty="l1".
    l1 = {"l1_ratio": 1.0} if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8) else {"penalty": "l1"}
    clf = LogisticRegression(C=0.1, solver="liblinear", max_iter=3000, random_state=RS, **l1)
    return Pipeline([("pre", pre), ("clf", clf)])


def entrenar(ruta_csv=DATA_CSV):
    """Entrena el modelo de producción con el 70 % del universo en mora y lo evalúa con el 30 %."""
    m = universo_en_mora(pd.read_csv(ruta_csv))
    y = 1 - m[OBJETIVO]  # PAGA
    Xtr, Xte, ytr, yte = train_test_split(m[VARIABLES], y, test_size=0.30, stratify=y, random_state=RS)
    modelo = construir_pipeline().fit(Xtr, ytr)
    p = modelo.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    metricas = {"auc": roc_auc_score(yte, p), "ks": float(np.max(tpr - fpr)), "n_train": len(Xtr), "n_test": len(Xte)}
    return modelo, metricas


def guardar(modelo, metricas, ruta=MODELO_PKL):
    import joblib
    ruta.parent.mkdir(exist_ok=True)
    joblib.dump({"modelo": modelo, "metricas": metricas, "sklearn": sklearn.__version__, "variables": VARIABLES}, ruta)


def cargar(ruta=MODELO_PKL):
    """Carga el .pkl; si no existe o fue creado con otra versión de scikit-learn, re-entrena (≈ 1 s)."""
    import joblib
    try:
        paquete = joblib.load(ruta)
        if paquete.get("sklearn") == sklearn.__version__:
            return paquete["modelo"], paquete["metricas"], "archivo .pkl"
    except Exception:
        pass
    modelo, metricas = entrenar()
    return modelo, metricas, "re-entrenado al iniciar"


def score(modelo, df: pd.DataFrame) -> np.ndarray:
    """Score de cobranzas 0–1000 = 1000 × probabilidad de pago."""
    return np.round(modelo.predict_proba(preparar(df)[VARIABLES])[:, 1] * 1000).astype(int)


def segmento(s: int):
    for nombre, minimo, accion in SEGMENTOS:
        if s >= minimo:
            return nombre, accion
