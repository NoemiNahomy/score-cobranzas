import io

import pandas as pd
import streamlit as st

import modelo as sc

st.set_page_config(page_title="Score de Cobranzas", page_icon="📊", layout="wide")


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
    df["SEGMENTO"] = df.SCORE.map(lambda s: sc.segmento(s)[0])
    df["ACCION_SUGERIDA"] = df.SCORE.map(lambda s: sc.segmento(s)[1])
    return df.sort_values("SCORE", ascending=False)


mdl, metricas, origen = obtener_modelo()

st.title("📊 Score de Cobranzas – clientes en mora")
st.caption("prueba con datos con ruido "
           f"Regresión logística · AUC {metricas['auc']:.3f} · KS {metricas['ks']:.3f} · modelo: {origen}")

tab_cli, tab_car, tab_data, tab_info = st.tabs(["👤 Cliente individual", "📁 Cartera (CSV)", "⬇️ Dataset de prueba", "ℹ️ Acerca del modelo"])

# ---------------------------------------------------------------- cliente individual
with tab_cli:
    meses = ["M0 · ago", "M1 · jul", "M2 · jun", "M3 · may", "M4 · abr", "M5 · mar"]
    c0, c1, c2, c3 = st.columns([1.1, 1, 1, 1])
    with c0:
        st.subheader("Cliente")
        lim = st.number_input("MTO_LIMITE_CREDITO (Bs)", 2_000, 300_000, 15_000, step=500)
        depto = st.selectbox("DEPARTAMENTO", sc.DEPARTAMENTOS)
        empleo = st.selectbox("TIPO_EMPLEO", sc.EMPLEOS)
    with c1:
        st.subheader("Estado de pago")
        st.caption("−2 sin consumo · −1 pagó total · 0 pagó mínimo · 1–8 meses de atraso")
        est = [st.number_input(f"EST_PAGO_{m.split(' ')[0]} ({m.split('· ')[1]})", -2, 8, 1 if i == 0 else 0, key=f"e{i}") for i, m in enumerate(meses)]
    with c2:
        st.subheader("Facturado (Bs)")
        fac = [st.number_input(f"MTO_FACTURADO_{m.split(' ')[0]}", -5_000, 300_000, 9_000, step=100, key=f"f{i}") for i, m in enumerate(meses)]
    with c3:
        st.subheader("Pagado (Bs)")
        pag = [st.number_input(f"MTO_PAGADO_{m.split(' ')[0]}", 0, 300_000, 500, step=50, key=f"p{i}") for i, m in enumerate(meses)]

    if est[0] < 1:
        st.info("El score está diseñado para clientes en mora (EST_PAGO_M0 ≥ 1).")
    if st.button("Calcular score", type="primary"):
        fila = pd.DataFrame([{"MTO_LIMITE_CREDITO": lim, "DEPARTAMENTO": depto, "TIPO_EMPLEO": empleo,
                              **dict(zip(sc.ESTADOS, est)), **dict(zip(sc.FACTURADOS, fac)), **dict(zip(sc.PAGADOS, pag))}])
        s = int(sc.score(mdl, fila)[0]); seg, accion = sc.segmento(s)
        k1, k2 = st.columns([1, 2])
        k1.metric("Score de cobranzas (0–1000)", s)
        k1.progress(s / 1000)
        k2.markdown(f"**Segmento:** {seg}  \n**Probabilidad estimada de pago:** {s / 10:.1f} %  \n**Acción sugerida:** {accion}")

# ---------------------------------------------------------------- cartera
with tab_car:
    st.write("Se califica la cartera del dataset de prueba. Solo se puntúan los clientes en mora (EST_PAGO_M0 ≥ 1).")
    datos = obtener_dataset()

    res = puntuar_cartera(datos)

    a, b, c = st.columns(3)
    a.metric("Clientes en mora", f"{len(res):,}".replace(",", "."))
    b.metric("Score promedio", int(res.SCORE.mean()))
    c.metric("Alta probabilidad de pago", f"{(res.SEGMENTO == sc.SEGMENTOS[0][0]).mean():.0%}")

    orden = [s[0] for s in sc.SEGMENTOS]
    resumen = (res.groupby("SEGMENTO")
                  .agg(clientes=("SCORE", "size"), score_promedio=("SCORE", "mean"))
                  .reindex(orden))
    if sc.OBJETIVO in res:
        resumen["tasa_pago_real"] = res.groupby("SEGMENTO")[sc.OBJETIVO].apply(lambda x: 1 - x.mean())

    st.dataframe(resumen.style.format({"score_promedio": "{:.0f}", "tasa_pago_real": "{:.1%}"}),
                 use_container_width=True)
    st.bar_chart(res.SEGMENTO.value_counts().reindex(orden))

    cols = [col for col in ["COD_CLIENTE", "SCORE", "SEGMENTO", "ACCION_SUGERIDA", "EST_PAGO_M0",
                            "MTO_LIMITE_CREDITO", "DEPARTAMENTO", "TIPO_EMPLEO"] if col in res]
    st.dataframe(res[cols].head(1000), use_container_width=True, hide_index=True)



# ---------------------------------------------------------------- dataset
with tab_data:
    st.write("**Cartera de tarjetas de crédito .** 30.025 registros con datos simulados "
             "con ruido aleatorio para esta prueba teórica; no corresponden a clientes reales. Montos en bolivianos (Bs), "
             "historial marzo–agosto 2026 (M5…M0) y variable objetivo en septiembre 2026.")
    d = obtener_dataset()
    st.dataframe(d.head(50), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)



# ---------------------------------------------------------------- info
with tab_info:
    st.markdown(f"""
**Modelo:** regresión logística con regularización L1 (C = 0,1), elegida entre 8 algoritmos por parsimonia e interpretabilidad.
Se usa la versión **sin variables demográficas** (sexo, educación, estado civil, edad).

**Desempeño en el conjunto de prueba** ({metricas['n_test']} clientes): AUC = {metricas['auc']:.3f} · KS = {metricas['ks']:.3f}.

**Score** = 1000 × probabilidad de pago del mes siguiente.

| Segmento | Score | Acción sugerida |
|---|---|---|
""" + "\n".join(f"| {n} | {r} | {a} |" for (n, _, a), r in zip(sc.SEGMENTOS, ["650–1000", "400–649", "0–399"])) + """

**Convención de nombres:** `COD_` código · `MTO_` monto en Bs · `EST_` estado · `IND_` indicador · `M0` mes de corte, `M1…M5` meses anteriores.

""")
