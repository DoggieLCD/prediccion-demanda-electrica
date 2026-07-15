import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import plotly.graph_objects as go
from prophet.plot import plot_components_plotly
import datetime

# 1. Configuración de página
st.set_page_config(page_title="Dashboard de Demanda", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ Dashboard de Predicción y Análisis de Demanda Sistémica")
st.markdown("Plataforma interactiva para visualizar el comportamiento histórico, predecir el consumo futuro y entender los patrones de estacionalidad.")

# 2. Funciones de Carga (Caché para optimizar rendimiento)
@st.cache_data
def load_data():
    df = pd.read_csv("demanda_sistemica_procesada.csv", sep=';', decimal=',', parse_dates=['fecha'])
    df = df[['fecha', 'demanda_sistemica_mw']].rename(columns={'fecha': 'ds', 'demanda_sistemica_mw': 'y'})
    return df.sort_values(by='ds')

@st.cache_resource
def load_model():
    with open('modelo_prophet_entrenado.json', 'r') as fin:
        return model_from_json(fin.read())

df_historico = load_data()
modelo = load_model()

# 3. Sidebar - Controles de Usuario
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")
    dias_prediccion = st.slider("Días a proyectar en el futuro", min_value=7, max_value=365, value=30, step=7)
    
    st.markdown("---")
    st.subheader("Filtro de Visualización")
    # Para no saturar el gráfico con datos desde 2006, permitimos filtrar el inicio
    anio_inicio = st.selectbox("Mostrar histórico desde:", [2020, 2021, 2022, 2023, 2024, 2025], index=4)

# 4. Generación de Predicción
# Creamos el dataframe futuro y predecimos
futuro = modelo.make_future_dataframe(periods=dias_prediccion)
prediccion = modelo.predict(futuro)

# 5. Tarjetas de KPIs (Dashboard Header)
st.markdown("### 📊 Indicadores Clave")
col1, col2, col3, col4 = st.columns(4)

ultimo_real = df_historico['y'].iloc[-1]
fecha_ultimo_real = df_historico['ds'].iloc[-1].strftime('%d-%m-%Y')

# Extraemos la predicción del periodo futuro
df_futuro = prediccion[prediccion['ds'] > df_historico['ds'].iloc[-1]]
peak_proyectado = df_futuro['yhat'].max()
fecha_peak = df_futuro.loc[df_futuro['yhat'].idxmax(), 'ds'].strftime('%d-%m-%Y')

col1.metric("Última Demanda Real", f"{ultimo_real:,.0f} MW", f"Registrado el {fecha_ultimo_real}")
col2.metric("Peak Máximo Proyectado", f"{peak_proyectado:,.0f} MW", f"Esperado el {fecha_peak}", delta_color="inverse")
col3.metric("Límite Superior (Riesgo)", f"{df_futuro['yhat_upper'].max():,.0f} MW", "Peor escenario")
col4.metric("Días Proyectados", f"+{dias_prediccion} días")

st.markdown("---")

# 6. Gráfico Principal: Histórico + Predicción + Rango
st.subheader("📈 Proyección de Demanda Sistémica y Rango de Confianza")

# Filtramos los datos para que el gráfico sea legible (desde el año seleccionado)
df_hist_filtrado = df_historico[df_historico['ds'].dt.year >= anio_inicio]
pred_filtrada = prediccion[prediccion['ds'].dt.year >= anio_inicio]

fig_main = go.Figure()

# Histórico
fig_main.add_trace(go.Scatter(
    x=df_hist_filtrado['ds'], y=df_hist_filtrado['y'], 
    mode='lines', name='Demanda Real', line=dict(color='#1f77b4', width=2)
))

# Rango Superior e Inferior (Banda de confianza)
fig_main.add_trace(go.Scatter(
    x=pd.concat([pred_filtrada['ds'], pred_filtrada['ds'][::-1]]),
    y=pd.concat([pred_filtrada['yhat_upper'], pred_filtrada['yhat_lower'][::-1]]),
    fill='toself', fillcolor='rgba(255, 127, 14, 0.2)', line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", showlegend=True, name='Rango de Incertidumbre'
))

# Línea de Predicción (yhat)
fig_main.add_trace(go.Scatter(
    x=pred_filtrada['ds'], y=pred_filtrada['yhat'], 
    mode='lines', name='Predicción (Tendencia)', line=dict(color='#ff7f0e', width=2, dash='dot')
))

fig_main.update_layout(
    template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0), height=500
)
st.plotly_chart(fig_main, use_container_width=True)

# 7. Explicación de los Resultados
with st.expander("📖 ¿Cómo interpretar este gráfico?", expanded=True):
    st.markdown("""
    * **Demanda Real (Línea Azul):** Representa los datos históricos registrados hasta la fecha actual.
    * **Predicción (Línea Naranja Punteada):** Es el valor esperado de demanda calculado por el algoritmo de Machine Learning (Prophet).
    * **Rango de Incertidumbre (Sombra Naranja):** Muestra los límites probables donde fluctuará la demanda. Si la línea real se sale de esta sombra, estamos ante una anomalía o *peak operacional inusual*.
    """)

st.markdown("---")

# 8. Análisis de Estacionalidad (Semanal y Anual)
st.subheader("🗓️ Desglose de Estacionalidad (Patrones de Consumo)")
st.markdown("El modelo detecta automáticamente patrones repetitivos. A continuación se desglosa cómo varía la demanda según el día y el mes.")

# Usamos la función nativa de Prophet para Plotly, que extrae exactamente lo que el modelo aprendió
fig_comp = plot_components_plotly(modelo, prediccion)
fig_comp.update_layout(template="plotly_white", autosize=True, height=700)

st.plotly_chart(fig_comp, use_container_width=True)

# Explicación de los componentes
col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    st.info("**Estacionalidad Semanal:** Muestra qué días de la semana tienen mayor o menor consumo energético. Típicamente los fines de semana muestran una caída respecto a los días hábiles.")
with col_exp2:
    st.info("**Tendencia y Feriados:** El algoritmo aísla el efecto de los días festivos y muestra si la demanda general del sistema está creciendo o decreciendo a largo plazo.")
