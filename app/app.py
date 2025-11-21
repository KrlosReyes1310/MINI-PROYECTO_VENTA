import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
import io

# --- Configuración de la Aplicación Streamlit ---
st.set_page_config(
    page_title="Análisis de Ventas y ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Rutas de Archivos (Siguiendo la estructura propuesta) ---
MODEL_PATH = 'models/modelo_regresion.pkl'
DATA_PATH = 'data/ventas.csv' # Simulación: Los datos se siguen generando si no se encuentran

# --- 1. Generación de Datos Sintéticos (Para asegurar la ejecución) ---
# En una aplicación real, aquí usarías pd.read_csv(DATA_PATH)

@st.cache_data
def load_data():
    """Genera un DataFrame de ventas sintético para la demostración."""
    
    # Intenta cargar datos si existieran (ej: en 'data/ventas.csv')
    # try:
    #     df = pd.read_csv(DATA_PATH)
    #     st.success(f"Datos cargados desde: {DATA_PATH}")
    #     return df
    # except FileNotFoundError:
    
    st.info("Generando datos sintéticos. En producción, usarías `pd.read_csv(DATA_PATH)` desde la carpeta `data/`.", icon="💡")

    np.random.seed(42)
    N = 5000
    
    # Columnas Categóricas
    regions = ['Norte', 'Sur', 'Este', 'Oeste']
    categories = ['Tecnología', 'Muebles', 'Material de Oficina']
    sales_array = np.random.lognormal(mean=5.5, sigma=1.2, size=N)
    
    data = {
        'Order_ID': [f'ID-{i:04d}' for i in range(N)],
        'Region': np.random.choice(regions, N, p=[0.25, 0.20, 0.35, 0.20]),
        'Category': np.random.choice(categories, N, p=[0.30, 0.35, 0.35]),
        'Sales': sales_array,
        'Discount': np.random.uniform(0.0, 0.4, N),
        'Profit': np.random.normal(loc=sales_array * 0.15, scale=sales_array * 0.3, size=N)
    }
    
    df = pd.DataFrame(data)
    
    # Limpieza básica: rellenar algunos valores nulos (simulados) y manejar duplicados
    df.loc[df.sample(frac=0.01).index, 'Profit'] = np.nan # Simular nulos
    df['Profit'].fillna(df['Profit'].median(), inplace=True)
    
    # Añadir un par de duplicados
    df_initial_len = len(df)
    df = pd.concat([df, df.iloc[[10, 20]]], ignore_index=True)
    df.drop_duplicates(subset=['Order_ID'], keep='first', inplace=True)
    
    # Guardar conteo inicial de nulos y duplicados en session_state para la Pestaña 1
    st.session_state.initial_len = df_initial_len
    st.session_state.initial_nulls = df['Profit'].isna().sum()
    
    return df

# --- 2. Título de la Aplicación ---

st.title("📊 Analizador de Datos de Ventas y Predictor de ML")
st.markdown("Plataforma interactiva para la exploración de datos, visualización y modelado predictivo de ventas.")

# Cargar los datos
df = load_data()

# --- 3. Definición de las Pestañas ---

tab1, tab2, tab3, tab4 = st.tabs(["1. Carga y Limpieza", "2. Análisis Exploratorio (EDA)", "3. Modelado Predictivo (ML)", "4. Predicción Interactiva"])

# ==============================================================================
# PESTAÑA 1: CARGA Y LIMPIEZA DE DATOS
# ==============================================================================

with tab1:
    st.header("Carga y Exploración Inicial del Dataset")
    
    col_data, col_info = st.columns([3, 2])

    with col_data:
        st.subheader("Datos de Ventas (Primeras 5 Filas)")
        st.dataframe(df.head())

        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df.describe().T)

    with col_info:
        st.subheader("Información del DataFrame (`info()` y Limpieza)")
        
        # Simular la salida de info()
        st.code(f"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: {len(df)} entries, 0 to {len(df)-1}
Data columns (total 6 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Order_ID  {len(df)} non-null    object 
 1   Region    {len(df)} non-null    object 
 2   Category  {len(df)} non-null    object 
 3   Sales     {len(df)} non-null    float64
 4   Discount  {len(df)} non-null    float64
 5   Profit    {len(df)} non-null    float64 
dtypes: float64(3), object(3)
memory usage: {df.memory_usage().sum() / 1024:.2f} KB
""")
        
        # Mostrar el resultado de la limpieza
        st.markdown(f"""
        - **Total de filas inicial (antes de duplicados):** {st.session_state.get('initial_len', len(df))}
        - **Filas después de eliminar duplicados (por Order_ID):** **{len(df)}**
        - **Valores Nulos en 'Profit' (rellenados con Mediana):** - `{st.session_state.get('initial_nulls', 0)}` valores nulos iniciales simulados.
            - **`0`** valores nulos después del relleno.
        """)

# ==============================================================================
# PESTAÑA 2: ANÁLISIS EXPLORATORIO (EDA)
# ==============================================================================

with tab2:
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    # 2.1 Ventas Totales por Región
    st.subheader("Ventas Totales por Región")
    sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=sales_by_region.index, y=sales_by_region.values, ax=ax1, palette="viridis")
    ax1.set_title('Ventas Totales por Región')
    ax1.set_xlabel('Región')
    ax1.set_ylabel('Ventas Totales')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    
    # 2.2 Ventas y Ganancias por Categoría (Productos más/menos rentables)
    st.subheader("Rendimiento por Categoría de Producto")
    category_summary = df.groupby('Category').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum')
    ).sort_values(by='Total_Sales', ascending=False)
    
    col_bar, col_scatter = st.columns(2)

    with col_bar:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        category_summary[['Total_Sales', 'Total_Profit']].plot(kind='bar', ax=ax2, secondary_y='Total_Profit', rot=0)
        ax2.set_title('Ventas y Ganancias por Categoría')
        ax2.set_xlabel('Categoría')
        st.pyplot(fig2)
        
        st.markdown(f"""
        **Insight Clave:** La categoría **{category_summary['Total_Profit'].idxmax()}** es la más rentable.
        """)
        
    with col_scatter:
        st.subheader("Relación entre Descuento y Ventas")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='Discount', y='Sales', hue='Category', data=df, alpha=0.6, ax=ax3)
        ax3.set_title('Ventas vs. Descuento')
        st.pyplot(fig3)


# ==============================================================================
# PESTAÑA 3: MODELADO PREDICTIVO (MACHINE LEARNING)
# ==============================================================================

# Cargar el modelo si existe, si no, crear el pipeline
@st.cache_resource
def load_model_pipeline():
    """Carga el modelo serializado o lo retorna como None."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as file:
                pipeline = pickle.load(file)
            st.success(f"Modelo cargado desde {MODEL_PATH} (¡Estructura de proyecto OK!)")
            return pipeline
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            return None
    return None

# Inicializar o cargar el modelo al inicio de la pestaña
model_pipeline = load_model_pipeline()

with tab3:
    st.header("Modelo de Regresión Lineal para Predicción de Ventas")
    st.markdown("""
    Objetivo: Predecir la **Venta (Sales)** usando el **Descuento (Discount)** y la **Categoría (Category)** como características.
    """)
    
    # 3.1 Preparación de Datos
    X = df[['Discount', 'Category']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.code(f"""
    Tamaño del conjunto de entrenamiento (X_train): {X_train.shape}
    Tamaño del conjunto de prueba (X_test): {X_test.shape}
    """)
    
    # Solo entrenar y guardar si el modelo no está cargado o si el usuario lo pide
    if model_pipeline is None or st.button("Re-entrenar y Guardar Modelo"):
        st.info("Entrenando un nuevo modelo y guardándolo en la carpeta `models/`.")
        
        # 3.2 Creación del Pipeline
        categorical_features = ['Category']
        numerical_features = ['Discount']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # 3.3 Entrenamiento del Modelo
        with st.spinner('Entrenando el modelo de Regresión Lineal...'):
            model_pipeline.fit(X_train, y_train)
        st.success("¡Modelo entrenado exitosamente!")
        
        # 3.3.1 Guardar el Modelo (Simulación de la estructura 'models/modelo_regresion.pkl')
        try:
            os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
            with open(MODEL_PATH, 'wb') as file:
                pickle.dump(model_pipeline, file)
            st.success(f"Modelo de Pipeline guardado en **{MODEL_PATH}**.")
        except Exception as e:
            st.warning(f"No se pudo guardar el modelo. Asegúrate de que la carpeta 'models/' exista. Error: {e}")
    
    # 3.4 Evaluación del Modelo (solo si el pipeline está disponible)
    if model_pipeline:
        y_pred = model_pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Métricas de Evaluación")
        
        col_metric, col_coef = st.columns(2)
        
        with col_metric:
            st.metric(label="Coeficiente de Determinación ($R^2$)", value=f"{r2:.4f}")
            st.markdown("""
            El valor $R^2$ (cercano a 1) indica qué tan bien el modelo predice la varianza de las ventas.
            """)

        with col_coef:
            st.subheader("Coeficientes del Modelo")
            try:
                # Mostrar coeficientes (esto requiere inspeccionar el pipeline)
                regressor = model_pipeline.named_steps['regressor']
                feature_names = numerical_features + list(model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
                coefficients = pd.Series(regressor.coef_, index=feature_names).sort_values(ascending=False)
                st.dataframe(coefficients.to_frame(name="Coeficiente"))
            except:
                st.info("Los coeficientes se mostrarán después de un entrenamiento exitoso.")

        # 3.5 Visualización de Predicciones vs Reales
        st.subheader("Predicciones del Modelo vs. Valores Reales (Conjunto de Prueba)")
        
        results = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
        results['Error'] = results['Real'] - results['Predicción']
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Real', y='Predicción', data=results, ax=ax4, alpha=0.6)
        max_val = max(results['Real'].max(), results['Predicción'].max())
        min_val = min(results['Real'].min(), results['Predicción'].min())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Perfecta')
        ax4.set_title('Valores Reales vs. Predicciones')
        st.pyplot(fig4)


# ==============================================================================
# PESTAÑA 4: PREDICCIÓN INTERACTIVA
# ==============================================================================

with tab4:
    st.header("Simulador de Predicción de Ventas")
    st.markdown("Ajusta los parámetros para obtener una predicción de ventas utilizando el modelo entrenado.")
    
    if model_pipeline is None:
        st.error("El modelo no ha sido entrenado o cargado. Por favor, ve a la Pestaña 3 para entrenarlo.")
    else:
        col_input, col_output = st.columns(2)
        
        with col_input:
            st.subheader("Parámetros de Entrada")
            
            # Entrada 1: Categoría (Selección)
            input_category = st.selectbox(
                "Seleccione la Categoría de Producto:",
                df['Category'].unique()
            )
            
            # Entrada 2: Descuento (Slider)
            input_discount = st.slider(
                "Seleccione el Descuento Aplicado:",
                min_value=0.0,
                max_value=0.4,
                value=0.1,
                step=0.01
            )
            
            input_data = pd.DataFrame({
                'Discount': [input_discount],
                'Category': [input_category]
            })
            
            if st.button("Calcular Predicción"):
                # Realizar la predicción
                try:
                    predicted_sales = model_pipeline.predict(input_data)[0]
                    st.session_state.predicted_sales = predicted_sales
                except Exception as e:
                    st.error(f"Error al predecir: {e}")
                    st.session_state.predicted_sales = None

        with col_output:
            st.subheader("Resultado de la Predicción")
            
            if 'predicted_sales' in st.session_state and st.session_state.predicted_sales is not None:
                st.metric(
                    label=f"Venta Esperada para '{input_category}' con {input_discount*100:.0f}% de Descuento",
                    value=f"${st.session_state.predicted_sales:,.2f}"
                )
                
                # Gráfico de la predicción
                avg_sales = df[df['Category'] == input_category]['Sales'].mean()

                fig5, ax5 = plt.subplots(figsize=(8, 4))
                data_to_plot = pd.Series([avg_sales, st.session_state.predicted_sales], index=['Venta Promedio', 'Predicción'])
                data_to_plot.plot(kind='barh', ax=ax5, color=['gray', 'darkorange'])
                ax5.set_title(f"Venta Promedio vs. Predicción para {input_category}")
                ax5.set_xlabel("Ventas ($)")
                st.pyplot(fig5)
                
            else:
                 st.info("Presione 'Calcular Predicción' para ver el resultado aquí.")


# --- Pie de página y Librerías Utilizadas ---
st.sidebar.title("📚 Librerías Usadas")
st.sidebar.markdown("""
- **Pandas** y **NumPy**: Manipulación de datos y generación sintética.
- **Matplotlib** y **Seaborn**: Visualizaciones en la pestaña EDA.
- **Scikit-learn**: Modelado predictivo (Regresión Lineal, Pipeline, Encoding).
- **Streamlit**: Despliegue de la aplicación web interactiva.
""")