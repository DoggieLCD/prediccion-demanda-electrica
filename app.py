#app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Configuración de la página
st.set_page_config(page_title="Predicción de Demanda Eléctrica", layout="wide")

st.title("⚡ Detección Dinámica de Peaks de Demanda Sistémica")
st.markdown("Plataforma interactiva del modelo predictivo basado en Prophet.")

# Cachear la carga de datos para que la app sea rápida
@st.cache_data
def load_data():
    df = pd.read_csv("demanda_sistemica_procesada.csv", sep=';', decimal=',', parse_dates=['fecha'])
    df = df[['fecha', 'demanda_sistemica_mw']].copy()
    df.rename(columns={'fecha': 'ds', 'demanda_sistemica_mw': 'y'}, inplace=True)
    df.sort_values(by='ds', inplace=True)
    return df

df_prophet = load_data()

# Cachear el entrenamiento del modelo
@st.cache_resource
def train_model(data):
    modelo = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    modelo.add_country_holidays(country_name='CL')
    modelo.fit(data)
    return modelo

with st.spinner('Entrenando el modelo predictivo...'):
    modelo = train_model(df_prophet)

# Predecir
dias_a_predecir = 270
futuro = modelo.make_future_dataframe(periods=dias_a_predecir, freq='D')
prediccion = modelo.predict(futuro)

# Filtro de meses para la visualización
st.sidebar.header("Configuración del Gráfico")
meses = st.sidebar.slider("Meses a visualizar en 2026", 1, 12, 9)

# Fechas dinámicas según el slider
fecha_inicio = '2026-01-01'
fecha_fin = f'2026-{meses:02d}-30' if meses != 2 else '2026-02-28'

df_plot_real = df_prophet[(df_prophet['ds'] >= fecha_inicio) & (df_prophet['ds'] <= fecha_fin)]
df_plot_pred = prediccion[(prediccion['ds'] >= fecha_inicio) & (prediccion['ds'] <= fecha_fin)]

# --- Renderizar el Gráfico ---
st.subheader(f"Análisis del Periodo: Enero - Mes {meses} (2026)")

fig, ax = plt.subplots(figsize=(12, 5))

# Zona de confianza
ax.fill_between(df_plot_pred['ds'], df_plot_pred['yhat_lower'], df_plot_pred['yhat_upper'], 
                 color='#fed03c', alpha=0.3, label='Zona de Confianza (95%)')

# Umbral y Demanda Real
ax.plot(df_plot_pred['ds'], df_plot_pred['yhat_upper'], color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7, label='Umbral de Peak')
ax.plot(df_plot_real['ds'], df_plot_real['y'], color='#333333', linewidth=1.2, label='Demanda Real')

# Peaks detectados
peaks = df_plot_real.merge(df_plot_pred[['ds', 'yhat_upper']], on='ds')
peaks_detectados = peaks[peaks['y'] > peaks['yhat_upper']]

if not peaks_detectados.empty:
    ax.scatter(peaks_detectados['ds'], peaks_detectados['y'], color='#fed03c', edgecolors='black', s=50, zorder=10, label='Eventos de Peak')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylabel('Demanda (MW)')
ax.legend(frameon=False, loc='upper right', ncol=2)

# Mostrar gráfico en Streamlit
st.pyplot(fig)
