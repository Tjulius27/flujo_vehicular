"""Prototipo Streamlit: serie histórica, pronóstico mensual y nivel de demanda.
Uso (desde la raíz del proyecto):  streamlit run src/app.py"""
import os
import pandas as pd
import streamlit as st
from modelos import (MODELOS, cargar_serie, limpiar_pandemia, nivel_de_demanda, pronosticar,
                     variacion_interanual)

st.set_page_config(page_title="Flujo vehicular en peajes", layout="wide")
st.title("Predicción del flujo vehicular en peajes")
st.caption("Fuente: INEI, Índice Nacional del Flujo Vehicular (base 2007 = 100), con datos de PROVIAS Nacional y OSITRÁN.")

RUTA = "data/indice_flujo_vehicular.csv"
nombres = {"total": "Total", "pesados": "Vehículos pesados", "ligeros": "Vehículos ligeros"}
serie_id = st.sidebar.selectbox("Serie", list(nombres), format_func=nombres.get)
modelo = st.sidebar.selectbox("Modelo", MODELOS, index=len(MODELOS) - 2)
horizonte = st.sidebar.slider("Meses a pronosticar", 1, 12, 6)
archivo = st.sidebar.file_uploader("Actualizar datos (CSV: fecha,total,pesados,ligeros)", type="csv")

if archivo is not None:
    pd.read_csv(archivo).to_csv("data/_subido.csv", index=False)
    ruta = "data/_subido.csv"
else:
    ruta = RUTA


@st.cache_data(show_spinner="Entrenando el modelo...")
def calcular(ruta, serie_id, modelo, horizonte):
    original = cargar_serie(ruta, serie_id)
    limpia = limpiar_pandemia(original)
    pron = pronosticar(modelo, limpia, horizonte)
    niveles, umbrales = nivel_de_demanda(limpia, pron)
    return original, pron, niveles, umbrales


original, pron, niveles, (u1, u2) = calcular(ruta, serie_id, modelo, horizonte)

c1, c2, c3 = st.columns(3)
c1.metric("Último mes registrado", f"{original.index[-1]:%m/%Y}", f"{original.iloc[-1]:.1f}")
c2.metric(f"Pronóstico {pron.index[0]:%m/%Y}", f"{pron.iloc[0]:.1f}")
c3.metric("Nivel de demanda del próximo mes", str(niveles.iloc[0]).upper())

grafico = pd.concat([original.rename("Histórico"),
                     pd.concat([original.iloc[[-1]], pron]).rename("Pronóstico")], axis=1)
st.subheader("Serie histórica y pronóstico")
st.line_chart(grafico)

st.subheader("Pronóstico por mes")
tabla = pd.DataFrame({"Mes": pron.index.strftime("%m/%Y"), "Índice pronosticado": pron.round(1).values,
                      "Nivel de demanda": niveles.astype(str).values})
st.dataframe(tabla, hide_index=True)
st.caption(f"Umbrales (terciles de la serie): bajo ≤ {u1:.1f} < medio ≤ {u2:.1f} < alto. "
           "El nivel describe la demanda vial, no la velocidad ni los tiempos de viaje.")

ruta_metricas = f"results/metricas_{serie_id}.csv"
if os.path.exists(ruta_metricas):
    st.subheader("Evaluación de los modelos (últimos 12 meses)")
    st.dataframe(pd.read_csv(ruta_metricas), hide_index=True)

with st.expander("Variación porcentual interanual (Ecuación 2)"):
    st.line_chart(variacion_interanual(original))