import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import calendar
import plotly.graph_objects as go

st.set_page_config(page_title="Predicción de Demanda", layout="wide")
st.title("⚡ Detección Dinámica de Peaks de Demanda Sistémica")
st.markdown("Plataforma interactiva del modelo predictivo basado en Prophet.")

# 1. Carga de datos históricos
@st.cache_data
def load_data():
    df = pd.read_csv("demanda_sistemica_procesada.csv", sep=';', decimal=',', parse_dates=['fecha'])
    df = df[['fecha', 'demanda_sistemica_mw']].copy()
    df.rename(columns={'fecha': 'ds', 'demanda_sistemica_mw': 'y'}, inplace=True)
    df.sort_values(by='ds', inplace=True)
    return df

df_prophet = load_data()

# 2. Carga del modelo pre-entrenado (¡Cero consumo de RAM!)
@st.cache_resource
def load_model():
    with open('modelo_prophet_entrenado.json', 'r') as fin:
        modelo_cargado = model_from_json(fin.read())
    return modelo_cargado

modelo = load_model()

# 3. Generación de predicciones cacheadas
@st.cache_data
def generate_predictions(_modelo, dias_a_predecir=270):
    futuro = _modelo.make_future_dataframe(periods=dias_a_predecir, freq='D')
    return _modelo.predict(futuro)

with st.spinner('Cargando predicciones...'):
    prediccion = generate_predictions(modelo, dias_a_predecir=270)

# ... [AQUÍ VA EXACTAMENTE EL MISMO CÓDIGO DEL SLIDER Y DE PLOTLY DEL MENSAJE ANTERIOR] ...
