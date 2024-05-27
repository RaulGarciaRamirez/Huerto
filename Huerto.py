import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
import datetime
import requests
from matplotlib.dates import MinuteLocator, DateFormatter
from streamlit_autorefresh import st_autorefresh

# Función para obtener y procesar datos de la última semana
@st.cache_data(ttl=60)
def fetch_data_from_api_last_month():
    now = datetime.datetime.now()
    one_month_ago = now - datetime.timedelta(days=30)
    one_month_ago_timestamp = int(one_month_ago.timestamp()) * 1000
    now_timestamp = int(now.timestamp()) * 1000
    url = "https://sensecap.seeed.cc/openapi/list_telemetry_data"
    auth = ('93I2S5UCP1ISEF4F', '6552EBDADED14014B18359DB4C3B6D4B3984D0781C2545B6A33727A4BBA1E46E')
    
    # Parámetros de la solicitud GET para el último mes
    params = {
        'device_eui': '2CF7F1C044300627',
        'time_start': one_month_ago_timestamp,
        'time_end': now_timestamp
    }

    response = requests.get(url, params=params, auth=auth)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error al realizar la solicitud. Código de estado: {response.status_code}")
        return None


def process_api_data(api_data):
    if not api_data:
        return pd.DataFrame(columns=['timestamp', 'CO2', 'temperature', 'humidity'])

    canal_identificacion = api_data['data']['list'][0]
    valores_medidos = api_data['data']['list'][1]

    data = {}
    for tipo_med, _ in zip(canal_identificacion, valores_medidos):
        channel_type = tipo_med[1]
        for valor, timestamp in _:
            fecha = pd.to_datetime(timestamp)
            valor = float(valor) if isinstance(valor, (int, float, str)) else None
            
            if fecha not in data:
                data[fecha] = {'timestamp': fecha}
            
            if channel_type == "4100":  # CO2
                data[fecha]['CO2'] = valor
            elif channel_type == "4098":  # Humedad
                data[fecha]['humidity'] = valor
            elif channel_type == "4097":  # Temperatura
                data[fecha]['temperature'] = valor

    df = pd.DataFrame.from_dict(data, orient='index')

    # Asegurarse de que los datos estén ordenados por timestamp
    df = df.sort_values(by='timestamp')
    
    # Eliminar duplicados
    df = df.drop_duplicates(subset='timestamp')
    
    return df

# Obtener y procesar los datos de la API del último mes
api_data_last_month = fetch_data_from_api_last_month()
df = process_api_data(api_data_last_month)

# Función para realizar DBSCAN
def run_dbscan(df):
    X = df[['CO2', 'temperature', 'humidity']].dropna()
    dbscan = DBSCAN(eps=1, min_samples=10).fit(X)
    df.loc[X.index, 'cluster'] = dbscan.labels_
    return df

# Función para realizar la regresión logística
def run_logistic_regression(df):
    threshold_temp = 21  # Umbral de temperatura para distinguir día y noche
    df['is_day'] = df['temperature'] > threshold_temp
    # Asegurarse de que haya al menos dos clases en los datos de destino
    if df['is_day'].nunique() < 2:
        st.error("No hay suficiente variabilidad en los datos para ajustar el modelo de regresión logística.")
        return df
    X = df[['temperature', 'humidity']].dropna()
    y = df['is_day']
    model = LogisticRegression()
    model.fit(X, y)
    df['day_pred'] = model.predict(X)
    return df


# Función para mostrar los datos crudos
def show_raw_data():
    st.subheader("Datos Crudos")
    st.write(df)

# Función para mostrar las gráficas de CO2, temperatura y humedad
def show_sensor_data():
    st.subheader("Gráficas de Sensores")
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    df_last_50 = df.tail(50)  # Obtener los últimos 50 registros

    # Gráfica de CO2
    df_last_50.plot(x='timestamp', y='CO2', ax=ax[0], title='CO2')
    ax[0].xaxis.set_major_locator(MinuteLocator(interval=5))  # Utilizar MinuteLocator
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    ax[0].set_xlabel('Timestamp')
    ax[0].set_ylabel('CO2')

    plt.subplots_adjust(hspace=0.9)  # Ajustar el espaciado vertical entre subgráficas
    plt.subplots_adjust(wspace=0.9)  # Ajustar el espaciado horizontal entre subgráficas

    # Gráfica de Temperatura
    df_last_50.plot(x='timestamp', y='temperature', ax=ax[1], title='Temperatura')
    ax[1].xaxis.set_major_locator(MinuteLocator(interval=5))  # Utilizar MinuteLocator
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    ax[1].set_xlabel('Timestamp')
    ax[1].set_ylabel('Temperatura')

    plt.subplots_adjust(hspace=0.9)  # Ajustar el espaciado vertical entre subgráficas
    plt.subplots_adjust(wspace=0.9)  # Ajustar el espaciado horizontal entre subgráficas

    # Gráfica de Humedad
    df_last_50.plot(x='timestamp', y='humidity', ax=ax[2], title='Humedad')
    ax[2].xaxis.set_major_locator(MinuteLocator(interval=5))  # Utilizar MinuteLocator
    ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    ax[2].set_xlabel('Timestamp')
    ax[2].set_ylabel('Humedad')

    st.pyplot(fig)

def show_analysis_data():
    st.subheader("Gráficas de Análisis")

    # Realizar DBSCAN
    df_dbscan = run_dbscan(df)

    # Realizar regresión logística
    df_regression = run_logistic_regression(df)
    
    # Obtener los últimos 50 registros
    df_dbscan_last_50 = df_dbscan.tail(100)
    df_regression_last_50 = df_regression.tail(100)

    # Gráficas de DBSCAN y Regresión Logística
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Gráfica de DBSCAN
    df_dbscan_last_50.plot(x='timestamp', y='CO2', ax=ax[0], title='DBSCAN', color='blue', linestyle='-')
    ax[0].scatter(df_dbscan_last_50[df_dbscan_last_50['cluster'] == -1]['timestamp'], df_dbscan_last_50[df_dbscan_last_50['cluster'] == -1]['CO2'], color='red')
    ax[0].xaxis.set_major_locator(MinuteLocator(interval=60))  # Utilizar MinuteLocator
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    ax[0].set_xlabel('Timestamp')
    ax[0].set_ylabel('CO2')

    # Indicador de presencia de gente en DBSCAN
    if df_dbscan['cluster'].nunique() > 1:
        st.text("Hay Gente: 🔴⚪")
    else:
        st.text("Hay Gente: ⚪🟢")

    plt.subplots_adjust(hspace=0.5)  # Ajustar el espaciado vertical entre subgráficas
    plt.subplots_adjust(wspace=0.5)  # Ajustar el espaciado horizontal entre subgráficas

    # Gráfica de Regresión Logística
    df_regression_last_50.plot(x='timestamp', y='temperature', ax=ax[1], title='Regresión Logística', color='yellow' if df_regression['day_pred'].mean() > 0.5 else 'navy', linestyle='None', marker='o', label='Día' if df_regression['day_pred'].mean() > 0.5 else 'Noche')
    ax[1].xaxis.set_major_locator(MinuteLocator(interval=60))  # Utilizar MinuteLocator
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    ax[1].set_xlabel('Timestamp')
    ax[1].set_ylabel('Temperatura')
    ax[1].legend()

    # Indicador de día/noche en Regresión Logística
    if df_regression['day_pred'].mean() > 0.5:
        st.text("Es de Día: ☀️")
    else:
        st.text("Es de Noche: 🌙")

    st.pyplot(fig)

# Aplicar el diseño con tabs
tabs = st.tabs(["Datos Crudos", "Gráficas de Sensores", "Gráficas de Análisis"])
with tabs[0]:
    show_raw_data()
with tabs[1]:
    show_sensor_data()
with tabs[2]:
    show_analysis_data()

# Añadir la función para actualizar automáticamente las gráficas cada minuto
st_autorefresh(interval=60000)