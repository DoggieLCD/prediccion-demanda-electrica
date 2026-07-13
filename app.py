import streamlit as st
import pandas as pd
from prophet import Prophet
import calendar
import plotly.graph_objects as go

# 1. Configuración principal de la página
st.set_page_config(page_title="Predicción de Demanda", layout="wide")
st.title("⚡ Detección Dinámica de Peaks de Demanda Sistémica")
st.markdown("Plataforma interactiva del modelo predictivo basado en Prophet. Pase el cursor sobre el gráfico para ver los valores exactos.")

# 2. Carga de datos
@st.cache_data
def load_data():
    # Leemos CSV procesado con separador ';' y coma decimal ','
    df = pd.read_csv("demanda_sistemica_procesada.csv", sep=';', decimal=',', parse_dates=['fecha'])
    # Seleccionamos y renombramos columnas para Prophet (ds, y)
    df = df[['fecha', 'demanda_sistemica_mw']].copy()
    df.rename(columns={'fecha': 'ds', 'demanda_sistemica_mw': 'y'}, inplace=True)
    df.sort_values(by='ds', inplace=True)
    return df

df_prophet = load_data()

# 3. Entrenamiento del modelo
@st.cache_resource
def train_model(data):
    # Creamos y entrenamos el modelo Prophet.
    modelo = Prophet(interval_width=0.95,
                     yearly_seasonality=False,
                     weekly_seasonality=True,
                     daily_seasonality=False)
    # Agregamos feriados de Chile
    modelo.add_country_holidays(country_name='CL')
    modelo.fit(data)
    return modelo

# Entrenamos el modelo (cache_resource evita reentrenar entre reruns)
with st.spinner('Entrenando modelo... (esto tomará unos segundos la primera vez)'):
    modelo = train_model(df_prophet)

# 4. Generación de predicciones
# NO cacheamos la función que recibe el objeto Prophet para evitar problemas de hashing.
def generate_predictions(modelo, dias_a_predecir=270):
    futuro = modelo.make_future_dataframe(periods=dias_a_predecir, freq='D')
    pred = modelo.predict(futuro)
    return pred

with st.spinner('Calculando predicciones...'):
    prediccion = generate_predictions(modelo, dias_a_predecir=270)

# 5. Barra interactiva en el panel lateral
st.sidebar.header("Configuración del Gráfico")
# Slider que selecciona el mes final de 2026 a mostrar (1..12)
mes_final = st.sidebar.slider("Mes a visualizar en 2026 (hasta)", 1, 12, 9)

# 6. Lógica de fechas
# Obtenemos el último día del mes seleccionado en 2026
ultimo_dia = calendar.monthrange(2026, mes_final)[1]
fecha_inicio = '2026-01-01'
fecha_fin = f'2026-{mes_final:02d}-{ultimo_dia:02d}'

inicio_dt = pd.to_datetime(fecha_inicio)
fin_dt = pd.to_datetime(fecha_fin)

# Filtramos datos reales y predicciones para el rango seleccionado
df_plot_real = df_prophet[(df_prophet['ds'] >= inicio_dt) & (df_prophet['ds'] <= fin_dt)]
df_plot_pred = prediccion[(prediccion['ds'] >= inicio_dt) & (prediccion['ds'] <= fin_dt)]

# 7. Renderizado del Gráfico Interactivo con Plotly
st.subheader(f"Análisis

