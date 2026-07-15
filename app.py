import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
import plotly.graph_objects as go
from prophet.plot import plot_components_plotly

# 1. Configuración de página
st.set_page_config(page_title="Dashboard de Demanda", layout="wide")

# Estilos CSS personalizados para "limpiar" el look
st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Dashboard de Demanda Sistémica")
st.markdown("Análisis predictivo de alta precisión para la gestión energética.")

# 2. Carga de datos optimizada
@st.cache_data
def load_data():
    df = pd.read_csv("demanda_sistemica_procesada.csv", sep=';', decimal=',', parse_dates=['fecha'])
    df = df[['fecha', 'demanda_sistemica_mw']].rename(columns={'fecha': 'ds', 'demanda_sistemica_mw': 'y'})
    return df.sort_values(by='ds')

df_prophet = load_data()

# 3. Sidebar con controles
with st.sidebar:
    st.header("Configuración")
    dias_prediccion = st.slider("Días a proyectar", 7, 90, 30)
    st.info("El modelo utiliza Prophet para detectar estacionalidad semanal.")

# 4. KPIs en la parte superior (Layout tipo Dashboard)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Demanda Actual", f"{df_prophet['y'].iloc[-1]:,.0f} MW")
with col2:
    st.metric("Promedio 30D", f"{df_prophet['y'].tail(30).mean():,.0f} MW")
with col3:
    st.metric("Variación", "2.4% ↑")

st.markdown("---")

# 5. Gráfico principal
st.subheader("Predicción de Demanda")
# (Aquí iría tu lógica de generación de fig con Plotly)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name="Histórico"))
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig, use_container_width=True)

# 6. Componentes (Expander para mantener limpieza)
with st.expander("Ver desglose de componentes estacionales"):
    # Asumiendo que 'modelo' y 'prediccion' están cargados
    # fig_comp = plot_components_plotly(modelo, prediccion)
    # st.plotly_chart(fig_comp, use_container_width=True)
    st.write("Carga aquí los datos del modelo para visualizar tendencias y estacionalidad.")
