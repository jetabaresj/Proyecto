import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib   
from tensorflow.keras.losses import MeanSquaredError


st.set_page_config(layout="wide")

st.markdown("""
<div style="text-align: center; font-size: 40px; font-weight: bold; ">
<i class="fas fa-search"></i> 
    ¡DETECCIÓN DE ANOMALÍAS EN CONSUMO DE GAS!
</div>
<div style="text-align: center; font-weight: bold; font-size: 24px;">      
    CLIENTES INDUSTRIALES DE CONTUGAS
</div>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)


def read_excel_file(filepath):
    """
    Lee un archivo Excel con múltiples hojas (una por cliente),
    y devuelve un único DataFrame combinado con una columna 'Cliente' que indica la hoja original.
    """
    excel = pd.ExcelFile(filepath)
    client_dfs = []

    for cliente in excel.sheet_names:
        df = pd.read_excel(excel, sheet_name=cliente)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')  # coerce para evitar errores con fechas mal formateadas
        df['Cliente'] = cliente
        client_dfs.append(df)

    data = pd.concat(client_dfs, ignore_index=True)
    return data



def prepare_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    # Configurar índice temporal para la descomposición
    df = df.copy()
    
    # Eliminar datos duplicados
    df = df.drop_duplicates(subset='Fecha')
    
    # Setear la fecha en el indice    
    df = df.set_index('Fecha').asfreq('h')
    
    # Interpolar valores faltantes
    df['Volumen'] = df['Volumen'].interpolate()
    
    return df

def extract_datetime_features(df: pd.DataFrame, fecha_col='Fecha') -> pd.DataFrame:
    # Copia para no modificar el original
    df_result = df.copy()
    
    if fecha_col not in df_result.columns:
        df_result[fecha_col] = df_result.index
    
    # Características básicas de tiempo
    df_result['hora'] = df_result[fecha_col].dt.hour
    df_result['dia'] = df_result[fecha_col].dt.day
    df_result['dia_semana'] = df_result[fecha_col].dt.dayofweek 
    df_result['dia_year'] = df_result[fecha_col].dt.dayofyear
    df_result['semana_year'] = df_result[fecha_col].dt.isocalendar().week
    df_result['mes'] = df_result[fecha_col].dt.month
    df_result['trimestre'] = df_result[fecha_col].dt.quarter
    df_result['year'] = df_result[fecha_col].dt.year
    
    # Características derivadas
    df_result['fin_de_semana'] = df_result['dia_semana'].isin([5, 6]).astype(int)
    df_result['dia_laboral'] = (~df_result['dia_semana'].isin([5, 6])).astype(int)
    
    # Festivos en Colombia
    festivos_colombia = holidays.country_holidays('CO', years=df_result[fecha_col].dt.year.unique())
    df_result['es_festivo'] = df_result[fecha_col].dt.date.isin(
        [d for d in festivos_colombia]
    ).astype(int)
        
    # Características cíclicas (seno y coseno)
    df_result['hora_seno'] = np.sin(2 * np.pi * df_result['hora'] / 24)
    df_result['hora_coseno'] = np.cos(2 * np.pi * df_result['hora'] / 24)
    df_result['dia_semana_seno'] = np.sin(2 * np.pi * df_result['dia_semana'] / 7)
    df_result['dia_semana_coseno'] = np.cos(2 * np.pi * df_result['dia_semana'] / 7)
    df_result['mes_seno'] = np.sin(2 * np.pi * df_result['mes'] / 12)
    df_result['mes_coseno'] = np.cos(2 * np.pi * df_result['mes'] / 12)
    
    return df_result  

def extract_seasonal_features(df: pd.DataFrame, period=24):
    # Crear copia para no modificar el original
    df_result = df.copy()
    
    if 'Fecha' not in df_result.columns:
        df_result['Fecha'] = df_result.index

    # Verificar frecuencia horaria
    if not isinstance(df_result.index, pd.DatetimeIndex):
        df_result = df_result.set_index('Fecha').asfreq('h')
    
    # Realizar descomposición estacional
    result = seasonal_decompose(df_result['Volumen'], model='additive', period=period)
    
    # Añadir componentes al DataFrame original
    df_result['trend'] = result.trend.values
    df_result['seasonal'] = result.seasonal.values
    df_result['residual'] = result.resid.values
    
    # Calcular características adicionales
    df_result['detrended'] = df_result['Volumen'] - df_result['trend']
    df_result['seas_strength'] = abs(df_result['seasonal'] / df_result['Volumen'])
    df_result['seas_norm'] = df_result['seasonal'] / df_result['seasonal'].std()
    
    return df_result

def calculate_rolling_features(df: pd.DataFrame, target_col='Volumen', windows=[24, 48, 168]):
    
    # Crear copia para no modificar el original
    df_result = df.copy()
    
    # Calcular estadísticas para cada ventana
    for window in windows:
        
        # Estadísticas móviles
        df_result[f'rolling_mean_{window}h'] = df_result[target_col].rolling(window=window, min_periods=1).mean()
        df_result[f'rolling_std_{window}h'] = df_result[target_col].rolling(window=window, min_periods=1).std()
        df_result[f'rolling_min_{window}h'] = df_result[target_col].rolling(window=window, min_periods=1).min()
        df_result[f'rolling_max_{window}h'] = df_result[target_col].rolling(window=window, min_periods=1).max()
        
        # Diferencias con respecto al promedio móvil
        df_result[f'diff_from_mean_{window}h'] = df_result[target_col] - df_result[f'rolling_mean_{window}h']
        df_result[f'pct_diff_from_mean_{window}h'] = (df_result[target_col] / df_result[f'rolling_mean_{window}h'] - 1) * 100
        
    # Lags
    for i in range(1,24):
        df_result[f'lag_{i}'] = df_result[target_col].shift(i)
        
    # Patrones semanales (168 horas)
    if 168 in windows:
        df_result['diff_from_last_week'] = df_result[target_col].diff(168)
        df_result['pct_diff_from_last_week'] = df_result[target_col].pct_change(168) * 100
    
    return df_result

def detect_outliers(df: pd.DataFrame, columns, method='zscore', threshold=3.0):
    
    # Crear copia para no modificar el original
    df_result = df.copy()
    
    for col in columns:
        if col not in df_result.columns:
            print(f"Columna {col} no encontrada en el DataFrame")
            continue
            
        if method == 'zscore':
            # Método Z-score
            z_scores = stats.zscore(df_result[col], nan_policy='omit')
            df_result[f'{col}_outlier_zscore'] = (abs(z_scores) > threshold).astype(int)
            
        elif method == 'iqr':
            # Método IQR
            Q1 = df_result[col].quantile(0.25)
            Q3 = df_result[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_result[f'{col}_outlier_iqr'] = ((df_result[col] < lower_bound) | 
                                               (df_result[col] > upper_bound)).astype(int)
    
    # Columna agregada de outliers
    outlier_cols = [col for col in df_result.columns if '_outlier_' in col]
    if outlier_cols:
        df_result['is_any_outlier'] = df_result[outlier_cols].max(axis=1)
    
    return df_result



def create_features_for_anomaly_detection(df: pd.DataFrame):
    """Función principal para crear todas las características para detección de anomalías"""
    # 0. Preparar los datos
    df_processed = prepare_hourly_data(df)
    
    # 1. Extraer características temporales
    df_processed = extract_datetime_features(df_processed)
    
    # 2. Añadir características de estacionalidad
    df_processed = extract_seasonal_features(df_processed)
    
    # 3. Calcular promedios móviles y estadísticas relacionadas
    df_processed = calculate_rolling_features(df_processed, target_col='Volumen')
            
    # 4. Detectar outliers en columnas relevantes
    columns_for_outlier_detection = ['Volumen']
    
    # Detectar outliers con ambos métodos
    df_processed = detect_outliers(df_processed, columns_for_outlier_detection, method='zscore')
    df_processed = detect_outliers(df_processed, columns_for_outlier_detection, method='iqr')
    
    return df_processed

# Código principal para procesar datos


def process_gas_data(data: pd.DataFrame):
    """Procesa datos de gas por cliente para detección de anomalías.
    Retorna una lista de DataFrames individuales por cliente.
    """
    results = []

    for cliente, df_cliente in data.groupby('Cliente'):
        print(f"\nProcesando cliente: {cliente}")
        
        # Aplicar todas las funciones de ingeniería de características
        df_processed = create_features_for_anomaly_detection(df_cliente)
        
        # Mostrar resumen de outliers detectados
        n_outliers = df_processed['is_any_outlier'].sum()
        print(f"Se detectaron {n_outliers} [{n_outliers / len(df_processed) * 100:.2f}%] posibles anomalías para el cliente {cliente}")
        
        # Guardar en CSV
        #df_processed.to_csv(f"/Users/canaveral/Downloads/{cliente}.csv", index=False)

        # Agregar a la lista de resultados
        results.append(df_processed)
    
    return results

# def process_gas_data(data: pd.DataFrame):
#     """Procesa datos de gas por cliente para detección de anomalías"""
#     results = {}
    
#     for cliente, df_cliente in data.groupby('Cliente'):
#         print(f"\nProcesando cliente: {cliente}")
        
#         # Aplicar todas las funciones de ingeniería de características
#         df_processed = create_features_for_anomaly_detection(df_cliente)
        
#         # Guardar resultados
#         results[cliente] = df_processed
        
#         # Mostrar resumen de outliers detectados
#         n_outliers = df_processed['is_any_outlier'].sum()
#         print(f"Se detectaron {n_outliers} [{n_outliers/len(df_processed) * 100:2f}%] posibles anomalías para el cliente {cliente}")
        
#         # Guardar en csv
#         df_processed.to_csv(f"/Users/canaveral/Downloads/{cliente}.csv", index=False)
    
#     return results

#data = pd.read_csv("/Users/canaveral/Downloads/data.csv", parse_dates=['Fecha'])

########### MODELOS DETECCION  OUTLIERS ###########
#########################

# --- FUNCIONES DE MODELOS ---
def detectar_con_isolation_forest(series_temporales, modelo_path='isolation_forest.pkl'):
    df = series_temporales.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df_modelo = df.drop(['Cliente', 'Fecha'], axis=1)

    scaler = RobustScaler()
    X = scaler.fit_transform(df_modelo)

    clf = joblib.load(modelo_path)
    predicciones = clf.predict(X)
    df['anomaly_isolation'] = np.where(predicciones == -1, 1, 0)

    return df

def detectar_con_autoencoder(series_temporales, modelo_path='autoencoder_model.h5', threshold_percentile=99):
    df = series_temporales.copy()
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    df_ae = df.drop(["Cliente", "Fecha"], axis=1)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_ae)

    autoencoder = load_model(modelo_path,  custom_objects={'mse': MeanSquaredError()})
    reconstructions = autoencoder.predict(X, verbose=0)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, threshold_percentile)

    df['anomaly_autoencoder'] = (mse > threshold).astype(int)
    return df

# --- PROCESAMIENTO GLOBAL DE CLIENTES ---
def procesar_clientes_con_modelos(results, nombres_clientes=None):
    results_process = []

    for i in range(1, 21):
        try:
            df = results[i - 1]
            cliente = nombres_clientes[i - 1] if nombres_clientes else f"CLIENTE{i}"
            original_rows = len(df)

            drop_cols = ['Volumen_outlier_zscore', 'is_any_outlier', 'Volumen_outlier_iqr']
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
            df = df.dropna()
            print(f"🧹 {cliente}: Se eliminaron {original_rows - len(df)} filas con NaN iniciales.")

            df = detectar_con_isolation_forest(df)
            print(f"🔁 {cliente}: Aplicado Isolation Forest.")

            df = detectar_con_autoencoder(df)
            print(f"📉 {cliente}: Aplicado Autoencoder.")

            # Nueva columna combinada
            df['is_any_outlier'] = df[['anomaly_isolation', 'anomaly_autoencoder']].max(axis=1)

            results_process.append((cliente, df))

        except Exception as e:
            print(f"❌ Error al procesar {cliente}: {e}")

    return results_process

# --- STREAMLIT APP ---
st.subheader("📁 Cargar archivo de Excel")
uploaded_file = st.file_uploader(".", type=['xlsx'])

st.markdown("---")
st.subheader("🔌 O conectarse a una Base de Datos SQL Server")

with st.expander("⚙️ Configuración de conexión a base de datos"):
    db_server = st.text_input("Servidor", value="localhost")
    db_name = st.text_input("Base de datos", value="nombre_basedatos")
    db_user = st.text_input("Usuario", value="sa")
    db_password = st.text_input("Contraseña", type="password")
    tabla = st.text_input("Nombre de la tabla de datos", value="ConsumoGas")

if st.button("📥 Cargar desde base de datos"):
    if all([db_server, db_name, db_user, db_password, tabla]):
        try:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={db_server};DATABASE={db_name};UID={db_user};PWD={db_password}"
            )
            conn = pyodbc.connect(conn_str)
            query = f"SELECT * FROM {tabla}"
            df_db = pd.read_sql(query, conn)

                        # 🔽 MOSTRAR POWER BI
            st.markdown("---")
            st.markdown("""
            <iframe title="Proyecto CONTUGAS" width="1024" height="1060"
            src="https://app.powerbi.com/view?r=eyJrIjoiMTUyYjYwNjEtY2FlYi00MGFhLTlmY2ItNGFkYmRkYzBlMDY3IiwidCI6ImU3OTg0Y2FjLTI1NDMtNGY4OC04Zjk3LTk1MjQzMzVlNmJjNCIsImMiOjR9"
            frameborder="0" allowFullScreen="true"></iframe>
            """, unsafe_allow_html=True)
            st.markdown("---")
            
        except Exception as e:
            st.error(f"❌ Error al conectar con la base de datos: {e}")
    else:
        st.warning("🔔 Completa todos los campos de conexión antes de continuar.")

# --- PROCESAMIENTO STREAMLIT ---

def new_func(uploaded_file):
    return uploaded_file

if uploaded_file is not None:
    with st.spinner('Procesando datos...'):
        # leer y procesar
        data = read_excel_file(new_func(uploaded_file))  # <-- debes tener esta función definida
        results = process_gas_data(data)                 # <-- debes tener esta función definida
        nombres_clientes = data['Cliente'].unique().tolist()
        procesados = procesar_clientes_con_modelos(results, nombres_clientes)
    st.success("✅ Datos procesados con éxito")
    st.markdown("---") 

    st.markdown("""
    <div style="position: relative; width: 100%; height: 0; padding-bottom: 130%;">
    <iframe title="Proyecto CONTUGAS"
            src="https://app.powerbi.com/view?r=eyJrIjoiMTUyYjYwNjEtY2FlYi00MGFhLTlmY2ItNGFkYmRkYzBlMDY3IiwidCI6ImU3OTg0Y2FjLTI1NDMtNGY4OC04Zjk3LTk1MjQzMzVlNmJjNCIsImMiOjR9"
            frameborder="0"
            allowFullScreen="true"
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
    </iframe>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")



# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from io import BytesIO
# import zipfile



# # --- Paso 1: Selector múltiple de clientes ---
# clientes_seleccionados = st.multiselect(
#     "Selecciona uno o más clientes para visualizar:",
#     [c for c, _ in procesados],
#     default=[c for c, _ in procesados][:1]
# )

# if clientes_seleccionados:
#     # --- Paso 2: Selector de rango de fechas ---
#     fechas_disponibles = pd.concat(
#         [df[['Fecha']] for c, df in procesados if c in clientes_seleccionados]
#     )
#     fecha_min = fechas_disponibles['Fecha'].min().date()
#     fecha_max = fechas_disponibles['Fecha'].max().date()

#     st.markdown("#### 📅 Selecciona un rango de fechas")
#     rango_fechas = st.date_input(
#         "Rango de fechas",
#         value=(fecha_min, fecha_max),
#         min_value=fecha_min,
#         max_value=fecha_max
#     )

#     if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
#         fecha_inicio = pd.to_datetime(rango_fechas[0])
#         fecha_fin = pd.to_datetime(rango_fechas[1])
#     else:
#         st.warning("Selecciona un rango válido de fechas.")
#         fecha_inicio, fecha_fin = pd.to_datetime(fecha_min), pd.to_datetime(fecha_max)

#     # --- Paso 3: Inicialización ---
#     total_anomalias = 0
#     total_registros = 0
#     datos_exportar = {}
#     resumen_tabla = []

#     fig = make_subplots(
#         rows=len(clientes_seleccionados), cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.05,
#         subplot_titles=clientes_seleccionados
#     )

#     # --- Paso 4: Recorrer clientes seleccionados ---
#     for idx, cliente in enumerate(clientes_seleccionados, start=1):
#         df_cliente = next(df for c, df in procesados if c == cliente)
#         df_cliente = df_cliente[
#             (df_cliente['Fecha'] >= fecha_inicio) & (df_cliente['Fecha'] <= fecha_fin)
#         ]

#         if 'Volumen' in df_cliente.columns and 'is_any_outlier' in df_cliente.columns:
#             normales = df_cliente[df_cliente['is_any_outlier'] == 0]
#             anomalias = df_cliente[df_cliente['is_any_outlier'] == 1]

#             n_anomalias = len(anomalias)
#             n_total = len(df_cliente)
#             pct_anomalias = (n_anomalias / n_total * 100) if n_total > 0 else 0

#             total_anomalias += n_anomalias
#             total_registros += n_total

#             resumen_tabla.append({
#                 "Cliente": cliente,
#                 "Total Registros": n_total,
#                 "Anomalías": n_anomalias,
#                 "% Anomalías": f"{pct_anomalias:.2f}%"
#             })

#             fig.add_trace(go.Scatter(
#                 x=normales['Fecha'], y=normales['Volumen'],
#                 mode='lines', name=f'{cliente} - Normal',
#                 line=dict(color='blue')
#             ), row=idx, col=1)

#             fig.add_trace(go.Scatter(
#                 x=anomalias['Fecha'], y=anomalias['Volumen'],
#                 mode='markers', name=f'{cliente} - Anomalía',
#                 marker=dict(color='red', size=6, symbol='x')
#             ), row=idx, col=1)

#             datos_exportar[cliente] = df_cliente.copy()
#         else:
#             st.warning(f"⚠️ {cliente} no tiene columnas 'Volumen' o 'is_any_outlier'.")

#     # --- Paso 5: Mostrar métricas justo después del título ---
#     st.markdown("### 📈 Conteo total de anomalías en el rango seleccionado")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Clientes Analizados", value=f"{len(clientes_seleccionados)}")
#     col2.metric("Total Registros", value=f"{total_registros:,}")
#     col3.metric("Total Anomalías", value=f"{total_anomalias:,}")
#     st.subheader("📊 Visualización de Anomalías por Cliente")
#     # --- Paso 6: Mostrar gráfica Plotly ---
#     fig.update_layout(
#         height=300 * len(clientes_seleccionados),
#         title_text="Consumo de Gas y Anomalías por Cliente",
#         showlegend=False,
#         template="plotly_white",
#         hovermode='x unified'
#     )
#     st.plotly_chart(fig, use_container_width=True)
        
#     correo_usuario = st.text_input("✉️ Ingresa tu correo electrónico para recibir el reporte")

#     if st.button("📧 Enviar reporte por correo"):
#         if "@" not in correo_usuario or "." not in correo_usuario:
#             st.error("Por favor ingresa un correo válido.")
   

# else:
#     st.info("Selecciona al menos un cliente para mostrar los resultados.")



      #Revisar!!!!    
    # Formulario de envío por email
    correo_usuario = st.text_input("✉️ Ingresa tu correo electrónico para recibir el reporte")

    if st.button("📧 Enviar reporte por correo"):
        if "@" not in correo_usuario or "." not in correo_usuario:
            st.error("Por favor ingresa un correo válido.")
   

#else:
#    st.info("Por favor sube un archivo Excel con hojas por cliente y columnas: Fecha, Volumen.")



