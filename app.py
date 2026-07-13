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
st.subheader(f"Análisis del Periodo: Enero - Mes {mes_final} (2026)")

fig = go.Figure()

# Zona de confianza (95%)
if not df_plot_pred.empty:
    fig.add_trace(go.Scatter(
        x=pd.concat([df_plot_pred['ds'], df_plot_pred['ds'][::-1]]),
        y=pd.concat([df_plot_pred['yhat_upper'], df_plot_pred['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(254, 208, 60, 0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Zona de Confianza (95%)',
        hoverinfo='skip'
    ))

    # Umbral de Peak (yhat_upper)
    fig.add_trace(go.Scatter(
        x=df_plot_pred['ds'],
        y=df_plot_pred['yhat_upper'],
        mode='lines',
        line=dict(color='#d62728', width=2, dash='dash'),
        name='Umbral de Peak'
    ))

# Demanda Real (serie histórica)
fig.add_trace(go.Scatter(
    x=df_plot_real['ds'],
    y=df_plot_real['y'],
    mode='lines',
    line=dict(color='#333333', width=2),
    name='Demanda Real'
))

# Detección de peaks: cuando la demanda real supera yhat_upper
if not df_plot_pred.empty and not df_plot_real.empty:
    peaks = df_plot_real.merge(df_plot_pred[['ds', 'yhat_upper']], on='ds', how='left')
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
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    margin=dict(l=20, r=20, t=20, b=20)
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
fig.update_xaxes(showgrid=False)

# 8. Mostramos el gráfico en Streamlit (container_width=True ajusta al ancho del layout)
st.plotly_chart(fig, use_container_width=True)
