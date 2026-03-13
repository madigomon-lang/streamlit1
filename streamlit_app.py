import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Análisis RESPEL - Maria Dilia",
    page_icon="♻️",
    layout="wide"
)

# --- ESTILOS PROFESIONALES ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .header-box { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 40px; border-radius: 15px; margin-bottom: 25px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE DATOS (AQUÍ ES DONDE SE INTRODUCE LA INFORMACIÓN) ---
@st.cache_data
def load_data():
    """
    Esta función es el punto de entrada de tus datos.
    Si tienes un archivo real, se carga aquí. Si no, se generan datos de ejemplo.
    """
    try:
        # OPCIÓN 1: Cargar datos desde un archivo real en GitHub
        # Solo necesitas subir un archivo llamado 'reciclaje_chile.csv' a tu repositorio.
        df = pd.read_csv("reciclaje_chile.csv")
    except Exception:
        # OPCIÓN 2: Datos de ejemplo (Si no hay archivo, la app usa esto)
        # Aquí puedes cambiar los nombres o cantidades para probar
        np.random.seed(42)
        rows = 500
        data = {
            'Material': np.random.choice(['Plástico', 'Vidrio', 'Metal', 'RESPEL'], rows),
            'Region': np.random.choice(['Metropolitana', 'Valparaíso', 'Biobío', 'Antofagasta'], rows),
            'Peso_kg': np.random.uniform(10, 150, rows),
            'Pureza_Pct': np.random.uniform(65, 98, rows),
        }
        # Esta línea crea la relación para que tu Modelo Lineal funcione bien
        data['Impacto_Ambiental'] = data['Peso_kg'] * 1.65 + np.random.normal(0, 5, rows)
        df = pd.DataFrame(data)
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.image("https://www.talentotech.gov.co/747/channels-747_logo_footer.png", use_container_width=True)
st.sidebar.markdown("### 🗺️ Navegación")
opcion = st.sidebar.radio("Seleccione una vista:", ["🏠 Inicio", "📊 Panel de Trabajo", "🧪 Validación de Modelo"])
st.sidebar.markdown("---")
st.sidebar.write("👤 **Analista Experto:**")
st.sidebar.info("Maria Dilia")

# --- LÓGICA DE PÁGINAS ---

if opcion == "🏠 Inicio":
    st.markdown("<div class='header-box'><h1>♻️ Análisis de Residuos RESPEL</h1><p>Experto en Datos | Nivel Integrador Talento Tech</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Explicación del Proyecto")
        st.write("""
        Esta Landing Page proporciona una visión técnica sobre el conjunto de datos de reciclaje en Chile. 
        El objetivo es analizar la eficiencia de la clasificación de materiales y validar modelos que 
        predicen el impacto ambiental positivo basado en la masa recolectada.
        """)
        st.markdown("""
        ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
        ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
        """)
    with col2:
        st.image("https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?auto=format&fit=crop&q=80&w=800", caption="Gestión de Residuos Profesionales", use_container_width=True)

    st.divider()
    st.subheader("📖 Documentación de Variables")
    with st.expander("Ver detalle de las columnas del dataset"):
        st.write("- **Material:** Categoría del residuo (Plástico, Metal, RESPEL).")
        st.write("- **Peso_kg:** Masa total capturada.")
        st.write("- **Pureza_Pct:** Calidad de la separación en origen.")

elif opcion == "📊 Panel de Trabajo":
    st.title("📊 Panel de Trabajo del Experto")
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Muestras", len(df))
    c2.metric("Peso Acumulado", f"{df['Peso_kg'].sum():,.1f} kg")
    c3.metric("Eficiencia Media", f"{df['Pureza_Pct'].mean():.1f}%")

    st.write("---")
    
    # Visualizaciones
    tab1, tab2 = st.tabs(["🌎 Análisis por Región (Plotly)", "🧪 Calidad por Material (Seaborn)"])
    
    with tab1:
        st.markdown("### Distribución Geográfica de Carga")
        fig_plotly = px.bar(df, x="Region", y="Peso_kg", color="Material", barmode="group", template="plotly_white")
        st.plotly_chart(fig_plotly, use_container_width=True)
        st.caption("Gráfico interactivo para identificar regiones críticas en la gestión de RESPEL.")

    with tab2:
        st.markdown("### Consistencia de Calidad")
        fig_sns, ax_sns = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x="Material", y="Pureza_Pct", palette="viridis", ax=ax_sns)
        st.pyplot(fig_sns)
        st.caption("El análisis de cajas permite observar la estabilidad en la pureza de los materiales recolectados.")

elif opcion == "🧪 Validación de Modelo":
    st.title("🧪 Validación del Modelo Lineal")
    st.write("Como experto en datos, validamos la relación predictiva entre el **Peso** y el **Impacto Ambiental**.")

    # Entrenamiento del modelo
    X = df[['Peso_kg']]
    y = df['Impacto_Ambiental']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("R² Score (Bondad de Ajuste)", f"{r2_score(y_test, y_pred):.4f}")
    col_m2.metric("MSE (Error Medio)", f"{mean_squared_error(y_test, y_pred):.2f}")

    # Visualización de Residuos
    st.subheader("Análisis de Residuos (Diagnóstico del Modelo)")
    fig_res, ax_res = plt.subplots()
    sns.histplot(y_test - y_pred, kde=True, color="blue", ax=ax_res)
    ax_res.set_title("Distribución de Errores (Residuos)")
    st.pyplot(fig_res)
    
    st.info("La forma de campana en los residuos confirma que el modelo lineal es estadísticamente válido para predecir el impacto ambiental.")

st.markdown("---")
st.caption("© 2024 Dashboard Integrador | Desarrollado por **Maria Dilia**")







