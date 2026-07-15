import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import plotly.graph_objects as go
from prophet.plot import plot_components_plotly

# ==========================================
# 1. CONFIGURACIÓN PRINCIPAL DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard de Demanda", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ Dashboard de Predicción y Análisis de Demanda Sistémica")
st.markdown("Plataforma interactiva para visualizar el comportamiento histórico, predecir el consumo futuro y entender los patrones de estacionalidad.")

# ==========================================
# 2. CARGA DE DATOS Y MODELO (Caché)
# ==========================================
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

# ==========================================
# 3. SIDEBAR - CONTROLES Y FILTROS
# ==========================================
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")
    dias_prediccion = st.slider("Días a proyectar en el futuro", min_value=7, max_value=365, value=30, step=7)
    
    st.markdown("---")
    st.subheader("Filtro de Visualización")
    # Opciones de años para no saturar el gráfico
    opciones_anios = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    anio_inicio = st.selectbox("Mostrar histórico desde el año:", opciones_anios, index=4) # Por defecto 2024

# ==========================================
# 4. GENERACIÓN DE PREDICCIÓN
# ==========================================
futuro = modelo.make_future_dataframe(periods=dias_prediccion)
prediccion = modelo.predict(futuro)

# ==========================================
# 5. TARJETAS DE INDICADORES (KPIs)
# ==========================================
st.markdown("### 📊 Indicadores Clave")
col1, col2, col3, col4 = st.columns(4)

ultimo_real = df_historico['y'].iloc[-1]
fecha_ultimo_real = df_historico['ds'].iloc[-1].strftime('%d-%m-%Y')

# Extraemos la predicción exclusivamente para el periodo futuro
df_futuro = prediccion[prediccion['ds'] > df_historico['ds'].iloc[-1]]
peak_proyectado = df_futuro['yhat'].max()
fecha_peak = df_futuro.loc[df_futuro['yhat'].idxmax(), 'ds'].strftime('%d-%m-%Y')

col1.metric("Última Demanda Real", f"{ultimo_real:,.0f} MW", f"Registrado el {fecha_ultimo_real}")
col2.metric("Peak Máximo Proyectado", f"{peak_proyectado:,.0f} MW", f"Esperado el {fecha_peak}", delta_color="inverse")
col3.metric("Límite Superior (Riesgo)", f"{df_futuro['yhat_upper'].max():,.0f} MW", "Peor escenario")
col4.metric("Días Proyectados", f"+{dias_prediccion} días")

st.markdown("---")

# ==========================================
# 6. GRÁFICO PRINCIPAL (Predicción solo en el futuro)
# ==========================================
st.subheader("📈 Proyección de Demanda Sistémica")

# Filtramos los datos según el año seleccionado
df_hist_filtrado = df_historico[df_historico['ds'].dt.year >= anio_inicio]
pred_filtrada = prediccion[prediccion['ds'].dt.year >= anio_inicio]

# Separamos el pasado del futuro para no mezclar líneas
ultima_fecha_real = df_hist_filtrado['ds'].max()
pred_futuro = pred_filtrada[pred_filtrada['ds'] > ultima_fecha_real]

fig_main = go.Figure()

# 1. Demanda Real Histórica (LÍNEA CONTINUA hasta el día de hoy)
fig_main.add_trace(go.Scatter(
    x=df_hist_filtrado['ds'], y=df_hist_filtrado['y'], 
    mode='lines', name='Demanda Real', 
    line=dict(color='#333333', width=2)
))

# 2. Predicción (Línea Azul, SÓLO HACIA EL FUTURO)
fig_main.add_trace(go.Scatter(
    x=pred_futuro['ds'], y=pred_futuro['yhat'], 
    mode='lines', name='Predicción Futura', 
    line=dict(color='#0072B2', width=2, dash='solid')
))

# 3. Zona de Riesgo (Sombra celeste, en todo el gráfico para ver anomalías pasadas y futuras)
fig_main.add_trace(go.Scatter(
    x=pd.concat([pred_filtrada['ds'], pred_filtrada['ds'][::-1]]),
    y=pd.concat([pred_filtrada['yhat_upper'], pred_filtrada['yhat_lower'][::-1]]),
    fill='toself', fillcolor='rgba(0, 114, 178, 0.2)', line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", showlegend=True, name='Intervalo de Confianza'
))

# 4. Detección de Peaks Operacionales (Tus círculos amarillos)
df_merge = pd.merge(df_hist_filtrado, pred_filtrada[['ds', 'yhat_upper']], on='ds', how='inner')
peaks = df_merge[df_merge['y'] > df_merge['yhat_upper']]

if not peaks.empty:
    fig_main.add_trace(go.Scatter(
        x=peaks['ds'], y=peaks['y'],
        mode='markers', name='Eventos de Peak',
        marker=dict(color='#fed03c', size=10, line=dict(color='black', width=1.5))
    ))

# 5. Diseño del gráfico principal
fig_main.update_layout(
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    margin=dict(l=0, r=0, t=30, b=0), height=500,
    yaxis_title="Demanda Sistémica (MW)",
    xaxis_title=""
)

# Línea vertical que marca "El Presente"
fig_main.add_vline(x=ultima_fecha_real, line_width=2, line_dash="dash", line_color="green", 
                   annotation_text="Hoy", annotation_position="top right")

st.plotly_chart(fig_main, use_container_width=True)

# ==========================================
# 7. ANÁLISIS DE ESTACIONALIDAD Y COMPONENTES
# ==========================================
st.subheader("🗓️ Desglose de Estacionalidad (Patrones de Consumo)")
st.markdown("El modelo detecta automáticamente patrones repetitivos. A continuación se desglosa cómo varía la demanda según la tendencia general y la estacionalidad periódica.")

# Gráficos interactivos de componentes de Prophet
fig_comp = plot_components_plotly(modelo, prediccion)
fig_comp.update_layout(template="plotly_white", autosize=True, height=700)

st.plotly_chart(fig_comp, use_container_width=True)

# Explicación de los componentes
col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    st.info("**Estacionalidad Semanal:** Muestra qué días de la semana tienen mayor o menor consumo energético. Típicamente los fines de semana muestran una caída respecto a los días hábiles.")
with col_exp2:
    st.info("**Tendencia y Feriados:** El algoritmo aísla el efecto de los días festivos y muestra si la demanda general del sistema está creciendo o decreciendo a largo plazo.")
