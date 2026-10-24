import pandas as pd
import streamlit as st

import modelo as sc

st.set_page_config(page_title="Score de Cobranzas", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- textos y colores
COLORES_SEG = ["#1E9E5A", "#E0A100", "#D64545"]  # mismo orden que sc.SEGMENTOS: alta, media, baja
RANGOS = ["650–1000", "400–649", "0–399"]

ESTADO_TXT = {-2: "Sin consumo", -1: "Pagó el total", 0: "Pagó el mínimo",
              **{k: f"{k} {'mes' if k == 1 else 'meses'} de atraso" for k in range(1, 9)}}
TXT_ESTADO = {v: k for k, v in ESTADO_TXT.items()}

NOMBRES_COL = {
    "COD_CLIENTE": "Cliente",
    "SCORE": "Score",
    "PROB_PAGO": "Prob. de pago",
    "SEGMENTO": "Segmento",
    "ACCION_SUGERIDA": "Acción sugerida",
    "EST_PAGO_M0": "Meses de atraso",
    "MTO_FACTURADO_M0": "Facturado ago. (Bs)",
    "MTO_LIMITE_CREDITO": "Límite de crédito (Bs)",
    "DEPARTAMENTO": "Departamento",
    "TIPO_EMPLEO": "Tipo de empleo",
}


def fmt(x: float) -> str:
    """Número con punto de miles (formato boliviano)."""
    return f"{x:,.0f}".replace(",", ".")


# ---------------------------------------------------------------- datos y modelo
@st.cache_resource(show_spinner="Cargando modelo…")
def obtener_modelo():
    return sc.cargar()


@st.cache_data(show_spinner="Leyendo dataset…")
def obtener_dataset():
    return pd.read_csv(sc.DATA_CSV)


def puntuar_cartera(df: pd.DataFrame) -> pd.DataFrame:
    faltan = [c for c in ["MTO_LIMITE_CREDITO", *sc.ESTADOS, *sc.FACTURADOS, *sc.PAGADOS, *sc.CATEGORICAS] if c not in df.columns]
    if faltan:
        st.error("Faltan columnas en el archivo: " + ", ".join(faltan)); st.stop()
    df = df[df.EST_PAGO_M0 >= 1].copy()
    df["SCORE"] = sc.score(mdl, df)
    df["PROB_PAGO"] = df.SCORE / 10
    df["SEGMENTO"] = df.SCORE.map(lambda s: sc.segmento(s)[0])
    df["ACCION_SUGERIDA"] = df.SCORE.map(lambda s: sc.segmento(s)[1])
    return df.sort_values("SCORE", ascending=False)


# ---------------------------------------------------------------- componentes visuales
def color_segmento(seg: str) -> str:
    nombres = [x[0] for x in sc.SEGMENTOS]
    return COLORES_SEG[nombres.index(seg)] if seg in nombres else "#888888"


def tarjeta_score(s: int, seg: str, accion: str) -> str:
    color = color_segmento(seg)
    pos = min(max(s / 10, 0), 100)
    de_cada_10 = round(s / 100)
    # El HTML va sin sangría ni líneas en blanco: si no, Streamlit lo interpreta como bloque de código
    return f"""
<div style="border:1px solid rgba(128,128,128,.25);border-left:6px solid {color};border-radius:14px;padding:20px 24px;margin-top:8px;">
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:28px;">
<div style="min-width:150px;">
<div style="font-size:.85rem;opacity:.7;">Score de cobranzas</div>
<div style="font-size:3.6rem;font-weight:700;line-height:1.05;color:{color};">{s}</div>
<div style="font-size:.8rem;opacity:.6;">de 1000 puntos</div>
</div>
<div style="flex:1;min-width:260px;">
<span style="background:{color};color:#fff;padding:5px 14px;border-radius:999px;font-size:.85rem;font-weight:600;">{seg}</span>
<div style="margin-top:12px;font-size:1.05rem;">Probabilidad de que pague el próximo mes: <b>{s / 10:.0f} %</b></div>
<div style="margin-top:4px;font-size:.95rem;opacity:.8;">De cada 10 clientes con un perfil parecido, unos <b>{de_cada_10}</b> pagarían.</div>
<div style="margin-top:10px;font-size:1rem;">👉 <b>Qué hacer:</b> {accion}</div>
</div>
</div>
<div style="position:relative;margin-top:26px;">
<div style="display:flex;height:12px;border-radius:6px;overflow:hidden;">
<div style="width:40%;background:{COLORES_SEG[2]};opacity:.35;"></div>
<div style="width:25%;background:{COLORES_SEG[1]};opacity:.35;"></div>
<div style="width:35%;background:{COLORES_SEG[0]};opacity:.35;"></div>
</div>
<div style="position:absolute;top:-5px;left:{pos}%;transform:translateX(-50%);width:6px;height:22px;border-radius:3px;background:{color};box-shadow:0 0 0 2px rgba(255,255,255,.8);"></div>
</div>
<div style="display:flex;font-size:.75rem;opacity:.65;margin-top:6px;">
<div style="width:40%;">0–399 · Baja</div>
<div style="width:25%;">400–649 · Media</div>
<div style="width:35%;text-align:right;">650–1000 · Alta</div>
</div>
</div>
"""


def tarjeta_segmento(nombre, rango, accion, n, pct, prob, monto=None, tasa=None) -> str:
    color = color_segmento(nombre)
    prob_txt = "—" if pd.isna(prob) else f"{prob:.0f} %"
    extra = ""
    if monto is not None:
        extra += f'<div style="margin-top:4px;font-size:.9rem;">Facturado en agosto: <b>Bs {fmt(monto)}</b></div>'
    if tasa is not None and not pd.isna(tasa):
        extra += f'<div style="margin-top:4px;font-size:.9rem;">Pagaron realmente (sep.): <b>{tasa:.0%}</b></div>'
    return f"""<div style="border:1px solid {color}55;background:{color}12;border-top:5px solid {color};border-radius:12px;padding:16px 18px;">
<div style="font-weight:700;font-size:1.05rem;color:{color};">{nombre}</div>
<div style="font-size:.78rem;opacity:.65;">Score {rango}</div>
<div style="font-size:2.2rem;font-weight:700;margin-top:8px;line-height:1.1;">{fmt(n)}</div>
<div style="font-size:.85rem;opacity:.75;">clientes · {pct:.0%} de la cartera en mora</div>
<div style="margin-top:10px;font-size:.9rem;">Probabilidad promedio de pago: <b>{prob_txt}</b></div>
{extra}
<div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(128,128,128,.25);font-size:.9rem;">👉 {accion}</div>
</div>"""


def barra_distribucion(partes: list[tuple[str, float]]) -> str:
    bloques = "".join(
        f'<div style="width:{p * 100:.2f}%;background:{color_segmento(nm)};color:#fff;font-size:.8rem;font-weight:600;'
        f'display:flex;align-items:center;justify-content:center;white-space:nowrap;overflow:hidden;">{p:.0%}</div>'
        for nm, p in partes if p > 0
    )
    return f'<div style="display:flex;height:30px;border-radius:8px;overflow:hidden;margin:4px 0 18px;">{bloques}</div>'


# ---------------------------------------------------------------- encabezado
mdl, metricas, origen = obtener_modelo()

st.title("📊 Score de Cobranzas")
st.markdown("##### ¿Qué clientes en mora tienen más probabilidad de pagar el próximo mes?")
st.caption("Datos con ruido")

tab_car, tab_cli, tab_info, tab_data = st.tabs(
    ["📋 Resumen de cartera", "👤 Evaluar un cliente", "ℹ️ Cómo leer el score", "🗂️ Datos"]
)

# ---------------------------------------------------------------- resumen de cartera
with tab_car:
    res = puntuar_cartera(obtener_dataset())
    orden = [s[0] for s in sc.SEGMENTOS]
    hay_monto = "MTO_FACTURADO_M0" in res
    hay_real = sc.OBJETIVO in res

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes en mora", fmt(len(res)))
    if hay_monto:
        k2.metric("Facturado en agosto (Bs)", fmt(res.MTO_FACTURADO_M0.sum()))
    k3.metric("Probabilidad promedio de pago", f"{res.PROB_PAGO.mean():.0f} %")
    k4.metric("Clientes a priorizar", fmt((res.SEGMENTO == orden[0]).sum()),
              help=f"Clientes del segmento «{orden[0]}» (score {RANGOS[0]}).")

    # Mensaje principal para gerencia
    alta = res[res.SEGMENTO == orden[0]]
    if len(alta):
        texto = (f"**Recomendación:** empezar la gestión por los **{fmt(len(alta))} clientes** del segmento "
                 f"**{orden[0]}**. Tienen en promedio **{alta.PROB_PAGO.mean():.0f} %** de probabilidad de pagar")
        if hay_monto:
            texto += f" y suman **Bs {fmt(alta.MTO_FACTURADO_M0.sum())}** facturados en agosto"
        st.success(texto + ".")

    st.markdown("#### ¿Cómo se reparte la cartera en mora?")
    st.markdown(barra_distribucion([(nm, (res.SEGMENTO == nm).mean()) for nm in orden]), unsafe_allow_html=True)

    cols_seg = st.columns(len(sc.SEGMENTOS))
    for col, (nombre, _, accion), rango in zip(cols_seg, sc.SEGMENTOS, RANGOS):
        sub = res[res.SEGMENTO == nombre]
        col.markdown(
            tarjeta_segmento(
                nombre, rango, accion,
                n=len(sub),
                pct=len(sub) / len(res) if len(res) else 0,
                prob=sub.PROB_PAGO.mean() if len(sub) else float("nan"),
                monto=sub.MTO_FACTURADO_M0.sum() if hay_monto else None,
                tasa=(1 - sub[sc.OBJETIVO].mean()) if hay_real and len(sub) else None,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("#### Lista de clientes priorizada")
    st.caption("Ordenada de mayor a menor probabilidad de pago. Se muestran hasta 1.000 clientes.")
    filtro = st.radio("Mostrar", ["Todos", *orden], horizontal=True, label_visibility="collapsed")
    vista = res if filtro == "Todos" else res[res.SEGMENTO == filtro]

    cols = [c for c in NOMBRES_COL if c in vista]
    st.dataframe(
        vista[cols].head(1000).rename(columns=NOMBRES_COL),
        use_container_width=True, hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1000, format="%d"),
            "Prob. de pago": st.column_config.NumberColumn("Prob. de pago", format="%.0f %%"),
            "Facturado ago. (Bs)": st.column_config.NumberColumn(format="%d"),
            "Límite de crédito (Bs)": st.column_config.NumberColumn(format="%d"),
        },
    )

# ---------------------------------------------------------------- evaluar un cliente
with tab_cli:
    st.write("Ingrese los datos del cliente y su historial de los últimos 6 meses para obtener su score.")

    izq, der = st.columns([1, 2.3], gap="large")
    with izq:
        st.subheader("Datos del cliente")
        lim = st.number_input("Límite de crédito (Bs)", 2_000, 300_000, 15_000, step=500)
        depto = st.selectbox("Departamento", sc.DEPARTAMENTOS)
        empleo = st.selectbox("Tipo de empleo", sc.EMPLEOS)

    with der:
        st.subheader("Historial de pagos")
        st.caption("Haga clic en una celda para cambiarla. Agosto es el mes actual (mes de corte).")
        historial = pd.DataFrame({
            "Mes": ["Agosto (actual)", "Julio", "Junio", "Mayo", "Abril", "Marzo"],
            "Situación de pago": [ESTADO_TXT[1]] + [ESTADO_TXT[0]] * 5,
            "Facturado (Bs)": [9_000] * 6,
            "Pagado (Bs)": [500] * 6,
        })
        editado = st.data_editor(
            historial, hide_index=True, use_container_width=True, num_rows="fixed",
            disabled=["Mes"], key="historial",
            column_config={
                "Situación de pago": st.column_config.SelectboxColumn(
                    options=list(ESTADO_TXT.values()), required=True,
                    help="Cómo terminó el cliente ese mes: sin consumo, pagó el total, pagó el mínimo o meses de atraso."),
                "Facturado (Bs)": st.column_config.NumberColumn(
                    min_value=-5_000, max_value=300_000, step=100, format="%d",
                    help="Monto del extracto del mes. Negativo = saldo a favor del cliente."),
                "Pagado (Bs)": st.column_config.NumberColumn(min_value=0, max_value=300_000, step=50, format="%d"),
            },
        )

    est = [TXT_ESTADO[v] for v in editado["Situación de pago"]]
    fac = editado["Facturado (Bs)"].fillna(0).tolist()
    pag = editado["Pagado (Bs)"].fillna(0).tolist()

    if est[0] < 1:
        st.info("Este cliente no está en mora en agosto. El score está pensado para clientes con al menos 1 mes de atraso.")
    if st.button("Calcular score", type="primary"):
        fila = pd.DataFrame([{"MTO_LIMITE_CREDITO": lim, "DEPARTAMENTO": depto, "TIPO_EMPLEO": empleo,
                              **dict(zip(sc.ESTADOS, est)), **dict(zip(sc.FACTURADOS, fac)), **dict(zip(sc.PAGADOS, pag))}])
        s = int(sc.score(mdl, fila)[0]); seg, accion = sc.segmento(s)
        st.markdown(tarjeta_score(s, seg, accion), unsafe_allow_html=True)

# ---------------------------------------------------------------- cómo leer el score
with tab_info:
    st.markdown("""
### ¿Qué es el score?
Es un número de **0 a 1000** que indica qué tan probable es que un cliente en mora **pague el próximo mes**.
Mientras más alto, más probable es el pago. Se lee directamente como porcentaje: un score de **720** equivale a
**72 %** de probabilidad de pago.

### ¿Para qué sirve?
Para **ordenar la gestión de cobranza**: dedicar llamadas y recursos primero a quienes tienen más
probabilidad de pagar, y reservar acciones más intensivas para los casos difíciles.
""")

    st.markdown("### Segmentos y acción recomendada")
    st.dataframe(
        pd.DataFrame([{"Segmento": n, "Score": r, "Acción sugerida": a}
                      for (n, _, a), r in zip(sc.SEGMENTOS, RANGOS)]),
        use_container_width=True, hide_index=True,
    )

    st.markdown(f"""
### ¿Qué tan confiable es?
Se probó con **{fmt(metricas['n_test'])} clientes** que el modelo no había visto antes.
Si se compara un cliente que pagó con uno que no pagó, el modelo le asigna un score más alto
al que pagó en el **{metricas['auc']:.0%} de los casos**.

### ¿Qué información usa?
El límite de crédito, el departamento, el tipo de empleo y el historial de los últimos 6 meses
(situación de pago, monto facturado y monto pagado). **No usa** sexo, educación, estado civil ni edad.
""")

    with st.expander("Detalle técnico"):
        st.markdown(f"""
- **Modelo:** regresión logística con regularización L1 (C = 0,1), elegida entre 8 algoritmos por parsimonia e interpretabilidad. Versión sin variables demográficas. Origen: {origen}.
- **Desempeño en prueba:** AUC = {metricas['auc']:.3f} · KS = {metricas['ks']:.3f}.
- **Score** = 1000 × probabilidad de pago del mes siguiente.
- **Convención de nombres:** `COD_` código · `MTO_` monto en Bs · `EST_` estado · `IND_` indicador · `M0` mes de corte, `M1…M5` meses anteriores.
""")

# ---------------------------------------------------------------- datos de prueba
with tab_data:
    st.write("**Cartera de tarjetas de crédito.** 30.025 registros con ruido aleatorio  esta prueba teórica-practica; "
             "no corresponden a clientes reales. Montos en bolivianos (Bs), historial marzo–agosto 2026 (M5…M0) "
             "y variable objetivo en septiembre 2026.")
    st.dataframe(obtener_dataset().head(50), use_container_width=True, hide_index=True)