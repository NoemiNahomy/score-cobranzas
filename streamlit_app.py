import unicodedata

import altair as alt
import pandas as pd
import pydeck as pdk
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


# Capitales de departamento (lat, lon) para ubicar cada departamento en el mapa
COORD_DEPTO = {
    "LA PAZ": (-16.4897, -68.1193), "SANTA CRUZ": (-17.7833, -63.1821), "COCHABAMBA": (-17.3895, -66.1568),
    "ORURO": (-17.9647, -67.1060), "POTOSI": (-19.5836, -65.7531), "CHUQUISACA": (-19.0196, -65.2619),
    "TARIJA": (-21.5355, -64.7296), "BENI": (-14.8333, -64.9000), "PANDO": (-11.0267, -68.7692),
}
MIN_GRUPO = 30  # grupos con menos clientes no se usan para conclusiones (porcentajes poco estables)


def normalizar(texto) -> str:
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().upper().strip()


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


# ---------------------------------------------------------------- análisis de perfil
def resumen_por(df: pd.DataFrame, col: str, seg: str) -> pd.DataFrame:
    """Por cada valor de `col`: clientes en mora, clientes del segmento, % del grupo y % del segmento."""
    g = (df.assign(_sel=df.SEGMENTO == seg)
           .groupby(col)
           .agg(en_mora=("SCORE", "size"), sel=("_sel", "sum"), prob=("PROB_PAGO", "mean")))
    g["pct"] = g.sel / g.en_mora
    g["part"] = g.sel / g.sel.sum() if g.sel.sum() else 0
    return g.sort_values("pct", ascending=False)


def grafico_barras(g: pd.DataFrame, titulo_cat: str, seg: str, color: str, promedio: float):
    d = g.reset_index().rename(columns={g.index.name: "cat"})
    d["etiqueta"] = [f"{p:.0%}  ({fmt(n)})" for p, n in zip(d.pct, d.sel)]
    orden_y = alt.EncodingSortField("pct", order="descending")
    base = alt.Chart(d).encode(y=alt.Y("cat:N", sort=orden_y, title=None))
    barras = base.mark_bar(color=color, cornerRadiusEnd=4).encode(
        x=alt.X("pct:Q", title=f"% de clientes en mora que son «{seg}»", axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip("cat:N", title=titulo_cat),
            alt.Tooltip("en_mora:Q", title="Clientes en mora", format=",.0f"),
            alt.Tooltip("sel:Q", title=f"Clientes «{seg}»", format=",.0f"),
            alt.Tooltip("pct:Q", title="% del grupo", format=".0%"),
            alt.Tooltip("part:Q", title="% del segmento", format=".0%"),
        ],
    )
    texto = base.mark_text(align="left", dx=5, fontSize=12).encode(x="pct:Q", text="etiqueta:N")
    regla = (alt.Chart(pd.DataFrame({"x": [promedio], "t": [f"Promedio cartera: {promedio:.0%}"]}))
               .mark_rule(strokeDash=[5, 4], color="gray").encode(x="x:Q", tooltip=alt.Tooltip("t:N", title="")))
    return (barras + texto + regla).properties(height=max(220, 36 * len(d)))


def hallazgo(g: pd.DataFrame, seg: str, promedio: float, articulo: str) -> str:
    validos = g[g.en_mora >= MIN_GRUPO]
    if validos.empty or g.sel.sum() == 0:
        return "No hay suficientes clientes para sacar conclusiones."
    top_pct, top_n = validos.pct.idxmax(), g.sel.idxmax()
    return (f"**Mayor riesgo relativo:** {articulo} **{top_pct}**, el {validos.pct[top_pct]:.0%} de los clientes en mora "
            f"es «{seg}» (promedio de la cartera: {promedio:.0%}).  \n"
            f"**Mayor volumen:** {articulo} **{top_n}** se concentran {fmt(g.sel[top_n])} clientes, "
            f"el {g.part[top_n]:.0%} de todo el segmento.")


def mapa_departamentos(g: pd.DataFrame, seg: str, color_hex: str):
    d = g.reset_index().rename(columns={g.index.name: "nombre"})
    d["clave"] = d.nombre.map(normalizar)
    d = d[d.clave.isin(COORD_DEPTO)].copy()
    if d.empty:
        return None
    d["lat"] = d.clave.map(lambda k: COORD_DEPTO[k][0])
    d["lon"] = d.clave.map(lambda k: COORD_DEPTO[k][1])
    maximo = max(d.sel.max(), 1)
    d["radio"] = 15_000 + 75_000 * (d.sel / maximo) ** 0.5           # tamaño = cantidad de clientes
    r, gg, b = (int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    pmax = max(d.pct.max(), 1e-9)
    d["color"] = [[r, gg, b, int(70 + 170 * p / pmax)] for p in d.pct]  # intensidad = % del departamento
    d["sel_txt"] = d.sel.map(fmt)
    d["pct_txt"] = d.pct.map(lambda p: f"{p:.0%}")
    d["mora_txt"] = d.en_mora.map(fmt)

    capas = [
        pdk.Layer("ScatterplotLayer", data=d, get_position=["lon", "lat"], get_radius="radio",
                  get_fill_color="color", get_line_color=[255, 255, 255], stroked=True,
                  line_width_min_pixels=1, pickable=True),
        pdk.Layer("TextLayer", data=d, get_position=["lon", "lat"], get_text="nombre", get_size=13,
                  get_color=[40, 40, 40], get_pixel_offset=[0, -22], font_weight=600),
    ]
    tooltip = {"html": f"<b>{{nombre}}</b><br/>Clientes «{seg}»: {{sel_txt}}<br/>"
                       "% de su cartera en mora: {pct_txt}<br/>Clientes en mora: {mora_txt}"}
    return pdk.Deck(layers=capas, tooltip=tooltip, map_style=None,
                    initial_view_state=pdk.ViewState(latitude=-16.6, longitude=-64.8, zoom=4.4))


def perfil_tipico(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    facturado = df[sc.FACTURADOS].clip(lower=0).sum().sum()
    pagado = df[sc.PAGADOS].sum().sum()
    return {
        "Clientes": fmt(len(df)),
        "Límite de crédito promedio (Bs)": fmt(df.MTO_LIMITE_CREDITO.mean()),
        "Facturado promedio en agosto (Bs)": fmt(df.MTO_FACTURADO_M0.mean()),
        "Meses de atraso en agosto (promedio)": f"{df.EST_PAGO_M0.mean():.1f}",
        "Meses con atraso en el semestre": f"{(df[sc.ESTADOS] >= 1).sum(axis=1).mean():.1f} de 6",
        "Pagó del total facturado (6 meses)": f"{pagado / facturado:.0%}" if facturado else "—",
    }


# ---------------------------------------------------------------- encabezado
mdl, metricas, origen = obtener_modelo()

st.title("📊 Score de Cobranzas")
st.markdown("##### ¿Qué clientes en mora tienen más probabilidad de pagar el próximo mes?")
st.caption("Prueba con datos ficticios (ruido aleatorio): no corresponden a clientes reales.")

tab_car, tab_perf, tab_cli, tab_info, tab_data = st.tabs(
    ["📋 Resumen de cartera", "🔎 Perfil de riesgo", "👤 Evaluar un cliente", "ℹ️ Cómo leer el score", "🗂️ Datos de prueba"]
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

# ---------------------------------------------------------------- perfil de riesgo
with tab_perf:
    st.write("¿Quiénes son los clientes con menor probabilidad de pago, dónde están y en qué trabajan?")
    seg_sel = st.selectbox("Segmento a analizar", orden, index=len(orden) - 1)
    color_sel = color_segmento(seg_sel)
    grupo = res[res.SEGMENTO == seg_sel]
    prom_cartera = (res.SEGMENTO == seg_sel).mean()

    p1, p2, p3 = st.columns(3)
    p1.metric(f"Clientes «{seg_sel}»", fmt(len(grupo)))
    p2.metric("% de la cartera en mora", f"{prom_cartera:.0%}")
    p3.metric("Probabilidad promedio de pago", f"{grupo.PROB_PAGO.mean():.0f} %" if len(grupo) else "—")

    # --- ubicación geográfica
    st.markdown("### 📍 ¿Dónde están?")
    g_depto = resumen_por(res, "DEPARTAMENTO", seg_sel)
    st.info(hallazgo(g_depto, seg_sel, prom_cartera, "En"))
    m1, m2 = st.columns([1, 1.15], gap="large")
    with m1:
        mapa = mapa_departamentos(g_depto, seg_sel, color_sel)
        if mapa is not None:
            st.pydeck_chart(mapa, use_container_width=True)
            st.caption("Tamaño del círculo = cantidad de clientes del segmento · Intensidad del color = % de la cartera en mora del departamento.")
        else:
            st.caption("No se pudieron ubicar los departamentos en el mapa.")
    with m2:
        st.altair_chart(grafico_barras(g_depto, "Departamento", seg_sel, color_sel, prom_cartera),
                        use_container_width=True)
        st.caption("La línea punteada es el promedio de toda la cartera en mora. Barras a la derecha de la línea = departamentos con más riesgo que el promedio.")

    # --- tipo de empleo
    st.markdown("### 💼 ¿En qué trabajan?")
    g_emp = resumen_por(res, "TIPO_EMPLEO", seg_sel)
    st.info(hallazgo(g_emp, seg_sel, prom_cartera, "Entre los clientes con empleo"))
    st.altair_chart(grafico_barras(g_emp, "Tipo de empleo", seg_sel, color_sel, prom_cartera),
                    use_container_width=True)

    # --- combinación departamento × empleo
    st.markdown("### 🧩 ¿Qué combinaciones concentran más riesgo?")
    st.caption(f"% de clientes «{seg_sel}» en cada combinación de departamento y tipo de empleo. "
               f"Se ocultan combinaciones con menos de {MIN_GRUPO} clientes en mora.")
    cruce = (res.assign(_sel=res.SEGMENTO == seg_sel)
                .groupby(["DEPARTAMENTO", "TIPO_EMPLEO"])
                .agg(en_mora=("SCORE", "size"), sel=("_sel", "sum"))
                .reset_index())
    cruce["pct"] = cruce.sel / cruce.en_mora
    cruce = cruce[cruce.en_mora >= MIN_GRUPO]
    if not cruce.empty:
        base = alt.Chart(cruce).encode(
            x=alt.X("TIPO_EMPLEO:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("DEPARTAMENTO:N", title=None),
        )
        calor = base.mark_rect(cornerRadius=3).encode(
            color=alt.Color("pct:Q", title="% del grupo", scale=alt.Scale(range=["#f6f6f6", color_sel]),
                            legend=alt.Legend(format="%")),
            tooltip=[alt.Tooltip("DEPARTAMENTO:N", title="Departamento"),
                     alt.Tooltip("TIPO_EMPLEO:N", title="Tipo de empleo"),
                     alt.Tooltip("en_mora:Q", title="Clientes en mora", format=",.0f"),
                     alt.Tooltip("sel:Q", title=f"Clientes «{seg_sel}»", format=",.0f"),
                     alt.Tooltip("pct:Q", title="% del grupo", format=".0%")],
        )
        valores = base.mark_text(fontSize=12).encode(
            text=alt.Text("pct:Q", format=".0%"),
            color=alt.condition(alt.datum.pct > cruce.pct.median(), alt.value("white"), alt.value("#333")),
        )
        st.altair_chart((calor + valores).properties(height=max(240, 38 * cruce.DEPARTAMENTO.nunique())),
                        use_container_width=True)
    else:
        st.caption("No hay combinaciones con suficientes clientes.")

    # --- perfil típico
    st.markdown("### 🧾 ¿Cómo es el cliente típico de cada segmento?")
    tabla = pd.DataFrame({nm: perfil_tipico(res[res.SEGMENTO == nm]) for nm in orden})
    tabla["Toda la cartera en mora"] = pd.Series(perfil_tipico(res))
    st.dataframe(tabla, use_container_width=True)
    st.caption("Comparar la columna del segmento con «Toda la cartera en mora» muestra qué lo diferencia del promedio.")

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
    st.write("**Cartera de tarjetas de crédito.** 30.025 registros con ruido aleatorio para esta prueba teórica; "
             "no corresponden a clientes reales. Montos en bolivianos (Bs), historial marzo–agosto 2026 (M5…M0) "
             "y variable objetivo en septiembre 2026.")
    st.dataframe(obtener_dataset().head(50), use_container_width=True, hide_index=True)