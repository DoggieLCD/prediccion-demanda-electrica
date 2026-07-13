import streamlit as st
import pandas as pd
import matplotlib
# Configuración estricta del motor gráfico para servidores sin pantalla (Evita el Segmentation fault)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prophet import Prophet
import calendar

# 1. Configuración principal de la página
st.set_page_config(page_title="Predicción de Demanda", layout="wide")

st.title("⚡ Detección Dinámica de Peaks de Demanda Sistémica")
st.markdown("Plataforma interactiva del modelo predictivo basado en Prophet.")

# 2. Carga de datos con caché para mayor velocidad
@st.cache_data
def load_data():
    df = pd.read_csv("demanda_sistemica_procesada.csv", sep=';', decimal=',', parse_dates=['fecha'])
    df = df[['fecha', 'demanda_sistemica_mw']].copy()
    df.rename(columns={'fecha': 'ds', 'demanda_sistemica_mw': 'y'}, inplace=True)
    df.sort_values(by='ds', inplace=True)
    return df

df_prophet = load_data()

# 3. Entrenamiento del modelo con caché
@st.cache_resource
def train_model(data):
    # Se ajusta yearly_seasonality=False para asegurar que la memoria RAM de Streamlit Cloud no colapse
    modelo = Prophet(interval_width=0.95, 
                     yearly_seasonality=False, 
                     weekly_seasonality=True, 
                     daily_seasonality=False)
    modelo.add_country_holidays(country_name='CL')
    modelo.fit(data)
    return modelo

with st.spinner('Entrenando el modelo predictivo...'):
    modelo = train_model(df_prophet)

# 4. Generación de predicciones
dias_a_predecir = 270
futuro = modelo.make_future_dataframe(periods=dias_a_predecir, freq='D')
prediccion = modelo.predict(futuro)

# 5. Barra interactiva en el panel lateral
st.sidebar.header("Configuración del Gráfico")
# Por defecto lo dejamos en 9, como solicitaste para el gráfico del póster
meses = st.sidebar.slider("Meses a visualizar en 2026", 1, 12, 9)

# 6. Lógica de fechas exactas y filtrado
ultimo_dia = calendar.monthrange(2026, meses)[1]
fecha_inicio = '2026-01-01'
fecha_fin = f'2026-{meses:02d}-{ultimo_dia:02d}'

inicio_dt = pd.to_datetime(fecha_inicio)
fin_dt = pd.to_datetime(fecha_fin)

df_plot_real = df_prophet[(df_prophet['ds'] >= inicio_dt) & (df_prophet['ds'] <= fin_dt)]
df_plot_pred = prediccion[(prediccion['ds'] >= inicio_dt) & (prediccion['ds'] <= fin_dt)]

# 7. Renderizado del Gráfico
st.subheader(f"Análisis del Periodo: Enero - Mes {meses} (2026)")

fig, ax = plt.subplots(figsize=(14, 6))

# Zona de confianza en amarillo
ax.fill_between(df_plot_pred['ds'], df_plot_pred['yhat_lower'], df_plot_pred['yhat_upper'], 
                 color='#fed03c', alpha=0.25, label='Zona de Confianza (95%)')

# Umbral de Peak (línea roja segmentada)
ax.plot(df_plot_pred['ds'], df_plot_pred['yhat_upper'], 
        color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7, label='Umbral de Peak')

# Demanda Real (línea principal gris oscura)
ax.plot(df_plot_real['ds'], df_plot_real['y'], 
        color='#333333', linewidth=1.2, label='Demanda Real')

# Puntos de Peaks detectados (amarillos con borde negro)
peaks = df_plot_real.merge(df_plot_pred[['ds', 'yhat_upper']], on='ds')
peaks_detectados = peaks[peaks['y'] > peaks['yhat_upper']]

if not peaks_detectados.empty:
    ax.scatter(peaks_detectados['ds'], peaks_detectados['y'], 
               color='#fed03c', edgecolors='black', s=60, zorder=10, label='Eventos de Peak')

# Configuración estética final
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylabel('Demanda Sistémica (MW)', fontsize=12)

# Centrado de leyenda
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=4)

plt.tight_layout()

# Mostrar gráfico en la web
st.pyplot(fig)

# 8. Liberación de memoria crítica
plt.close(fig)

