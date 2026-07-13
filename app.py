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
    df = pd.read_csv("demanda_sistemica_procesada.csv", sep=';', decimal=',', parse_dates=['fecha'])
    df = df[['fecha', 'demanda_sistemica_mw']].copy()
    df.rename(columns={'fecha': 'ds', 'demanda_sistemica_mw': 'y'}, inplace=True)
    df.sort_values(by='ds', inplace=True)
    return df

df_prophet = load_data()

# 3. Entrenamiento del modelo
@st.cache_resource
def train_model(data):
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
meses = st.sidebar.slider("Meses a visualizar en 2026", 1, 12, 9)

# 6. Lógica de fechas
ultimo_dia = calendar.monthrange(2026, meses)[1]
fecha_inicio = '2026-01-01'
fecha_fin = f'2026-{meses:02d}-{ultimo_dia:02d}'

inicio_dt = pd.to_datetime(fecha_inicio)
fin_dt = pd.to_datetime(fecha_fin)

df_plot_real = df_prophet[(df_prophet['ds'] >= inicio_dt) & (df_prophet['ds'] <= fin_dt)]
df_plot_pred = prediccion[(prediccion['ds'] >= inicio_dt) & (prediccion['ds'] <= fin_dt)]

# 7. Renderizado del Gráfico Interactivo con Plotly
st.subheader(f"Análisis del Periodo: Enero - Mes {meses}")

fig = go.Figure()

# Zona de confianza (Relleno amarillo)
fig.add_trace(go.Scatter(
    x=pd.concat([df_plot_pred['ds'], df_plot_pred['ds'][::-1]]),
    y=pd.concat([df_plot_pred['yhat_upper'], df_plot_pred['yhat_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(254, 208, 60, 0.25)', # #fed03c con 25% de opacidad
    line=dict(color='rgba(255,255,255,0)'),
    name='Zona de Confianza (95%)',
    showlegend=True,
    hoverinfo='skip'
))

# Umbral de Peak (Línea roja segmentada)
fig.add_trace(go.Scatter(
    x=df_plot_pred['ds'],
    y=df_plot_pred['yhat_upper'],
    mode='lines',
    line=dict(color='#d62728', width=2, dash='dash'),
    name='Umbral de Peak'
))

# Demanda Real (Línea principal gris)
fig.add_trace(go.Scatter(
    x=df_plot_real['ds'],
    y=df_plot_real['y'],
    mode='lines',
    line=dict(color='#333333', width=2),
    name='Demanda Real'
))

# Puntos de Peaks detectados
peaks = df_plot_real.merge(df_plot_pred[['ds', 'yhat_upper']], on='ds')
peaks_detectados = peaks[peaks['y'] > peaks['yhat_upper']]

if not peaks_detectados.empty:
    fig.add_trace(go.Scatter(
        x=peaks_detectados['ds'],
        y=peaks_detectados['y'],
        mode='markers',
        marker=dict(color='#fed03c', size=10, line=dict(color='black', width=1.5)),
        name='Eventos de Peak'
    ))

# Diseño del gráfico
fig.update_layout(
    yaxis_title="Demanda Sistémica (MW)",
    xaxis_title="",
    plot_bgcolor='white',
    hovermode="x unified", # Muestra una línea vertical con todos los datos al pasar el mouse
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=20, r=20, t=20, b=20)
)

# Agregar la grilla tipo matplotlib
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
fig.update_xaxes(showgrid=False)

# Mostrar en Streamlit
st.plotly_chart(fig, use_container_width=True)

