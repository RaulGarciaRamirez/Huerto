import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
import datetime
import requests
from matplotlib.dates import MinuteLocator, DateFormatter
from streamlit_autorefresh import st_autorefresh

# Función para obtener y procesar datos de la API
@st.cache_data(ttl=60)
def fetch_data_from_api(time_start, time_end):
    url = "https://sensecap.seeed.cc/openapi/list_telemetry_data"
    auth = ('93I2S5UCP1ISEF4F', '6552EBDADED14014B18359DB4C3B6D4B3984D0781C2545B6A33727A4BBA1E46E')

    params = {
        'device_eui': '2CF7F1C044300627',
        'time_start': time_start,
        'time_end': time_end
    }

    response = requests.get(url, params=params, auth=auth)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error al realizar la solicitud. Código de estado: {response.status_code}")
        return None

def process_api_data(api_data):
    if not api_data or 'data' not in api_data or 'list' not in api_data['data']:
        st.error("Datos de la API no válidos o incompletos.")
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

# Función para realizar DBSCAN
def run_dbscan(df):
    X = df[['CO2', 'temperature', 'humidity']].dropna()
    if time_range == "Últimas 24 horas":
        dbscan = DBSCAN(eps=3, min_samples=10).fit(X)
    elif time_range == "Última semana":
        dbscan = DBSCAN(eps=3, min_samples=15).fit(X)
    else:  # Último mes
        dbscan = DBSCAN(eps=3, min_samples=50).fit(X)
    df.loc[X.index, 'cluster'] = dbscan.labels_
    return df

# Función para mostrar los datos crudos
def show_raw_data():
    st.subheader("Datos Crudos")
    st.write("""
        En esta sección, se muestran los datos crudos obtenidos de los sensores. Los datos incluyen mediciones de CO2, temperatura y humedad con sus respectivas marcas de tiempo. 
        Estos datos son la base para todo el análisis posterior.
    """)
    st.write(df)

# Función para mostrar las gráficas de CO2, temperatura y humedad
def show_sensor_data():
    st.subheader("Gráficas de Sensores")
    st.write("""
        Aquí se muestran las gráficas de las mediciones de CO2, temperatura y humedad obtenidas de los sensores en el rango de tiempo seleccionado.
        Estas gráficas nos permiten visualizar las tendencias y cambios en las mediciones a lo largo del tiempo.
    """)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # Gráfica de CO2
    if time_range == "Últimas 24 horas":
        ax[0].xaxis.set_major_locator(MinuteLocator(interval=60))  # Utilizar MinuteLocator con intervalo de 60 minutos
        ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    elif time_range == "Última semana":
        ax[0].xaxis.set_major_locator(MinuteLocator(interval=1440))  # Utilizar MinuteLocator con intervalo de 1440 minutos (1 día)
        ax[0].xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    else:  # Último mes
        ax[0].xaxis.set_major_locator(MinuteLocator(interval=4320))  # Utilizar MinuteLocator con intervalo de 4320 minutos (3 días)
        ax[0].xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    df.plot(x='timestamp', y='CO2', ax=ax[0], title='CO2')
    ax[0].set_xlabel('Timestamp')
    ax[0].set_ylabel('CO2')

    plt.subplots_adjust(hspace=0.9)  # Ajustar el espaciado vertical entre subgráficas
    plt.subplots_adjust(wspace=0.9)  # Ajustar el espaciado horizontal entre subgráficas

    # Gráfica de Temperatura
    if time_range == "Últimas 24 horas":
        ax[1].xaxis.set_major_locator(MinuteLocator(interval=60))  # Utilizar MinuteLocator con intervalo de 60 minutos
        ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    elif time_range == "Última semana":
        ax[1].xaxis.set_major_locator(MinuteLocator(interval=1440))  # Utilizar MinuteLocator con intervalo de 1440 minutos (1 día)
        ax[1].xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    else:  # Último mes
        ax[1].xaxis.set_major_locator(MinuteLocator(interval=4320))  # Utilizar MinuteLocator con intervalo de 4320 minutos (3 días)
        ax[1].xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    df.plot(x='timestamp', y='temperature', ax=ax[1], title='Temperatura')
    ax[1].set_xlabel('Timestamp')
    ax[1].set_ylabel('Temperatura')

    plt.subplots_adjust(hspace=0.9)  # Ajustar el espaciado vertical entre subgráficas
    plt.subplots_adjust(wspace=0.9)  # Ajustar el espaciado horizontal entre subgráficas

    # Gráfica de Humedad
    if time_range == "Últimas 24 horas":
        ax[2].xaxis.set_major_locator(MinuteLocator(interval=60))  # Utilizar MinuteLocator con intervalo de 60 minutos
        ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    elif time_range == "Última semana":
        ax[2].xaxis.set_major_locator(MinuteLocator(interval=1440))  # Utilizar MinuteLocator con intervalo de 1440 minutos (1 día)
        ax[2].xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    else:  # Último mes
        ax[2].xaxis.set_major_locator(MinuteLocator(interval=4320))  # Utilizar MinuteLocator con intervalo de 4320 minutos (3 días)
        ax[2].xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    df.plot(x='timestamp', y='humidity', ax=ax[2], title='Humedad')
    ax[2].set_xlabel('Timestamp')
    ax[2].set_ylabel('Humedad')

    st.pyplot(fig)

def show_analysis_data():
    st.subheader("Gráficas de Análisis")
    st.write("""
        En esta sección, se utiliza el algoritmo de agrupamiento DBSCAN para identificar patrones y posibles presencias de personas según los datos de CO2.
        Los puntos rojos en la gráfica indican posibles presencias de personas o anomalías.
        Si el indicador de gente está en rojo 🔴⚪ significa que hay gente en el huerto, si está en verde ⚪🟢 significa que no hay gente en el huerto.
    """)
    # Realizar DBSCAN
    df_dbscan = run_dbscan(df)

    # Gráfica de DBSCAN
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Gráfica de DBSCAN
    if time_range == "Últimas 24 horas":
        ax.xaxis.set_major_locator(MinuteLocator(interval=60))  # Utilizar MinuteLocator con intervalo de 1 hora
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))  # Formatear el eje x como hora:minuto
    elif time_range == "Última semana":
        ax.xaxis.set_major_locator(MinuteLocator(interval=1440))  # Utilizar MinuteLocator con intervalo de 1440 minutos (1 día)
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    else:  # Último mes
        ax.xaxis.set_major_locator(MinuteLocator(interval=4320))  # Utilizar MinuteLocator con intervalo de 4320 minutos (3 días)
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))  # Formatear el eje x como mes-día
    df_dbscan.plot(x='timestamp', y='CO2', ax=ax, title='DBSCAN', color='blue', linestyle='-')
    ax.scatter(df_dbscan[df_dbscan['cluster'] == -1]['timestamp'], df_dbscan[df_dbscan['cluster'] == -1]['CO2'], color='red')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('CO2')

    # Indicador de presencia de gente en DBSCAN
    if df_dbscan['cluster'].nunique() > 1:
        st.text("Gente en el huerto: 🔴⚪")
    else:
        st.text("Gente en el huerto: ⚪🟢")

    plt.subplots_adjust(hspace=0.5)  # Ajustar el espaciado vertical entre subgráficas
    plt.subplots_adjust(wspace=0.5)  # Ajustar el espaciado horizontal entre subgráficas

    st.pyplot(fig)

# Añadir el desplegable para seleccionar el rango de tiempo
time_range = st.selectbox("Seleccione el rango de tiempo para ver los datos", ["Últimas 24 horas", "Última semana", "Último mes"])

# Obtener las marcas de tiempo según la selección
now = datetime.datetime.now()
if time_range == "Últimas 24 horas":
    start_time = now - datetime.timedelta(days=1)
elif time_range == "Última semana":
    start_time = now - datetime.timedelta(weeks=1)
else:  # Último mes
    start_time = now - datetime.timedelta(days=30)

start_timestamp = int(start_time.timestamp()) * 1000
end_timestamp = int(now.timestamp()) * 1000

# Obtener y procesar los datos de la API según el rango de tiempo seleccionado
api_data = fetch_data_from_api(start_timestamp, end_timestamp)
df = process_api_data(api_data)

# Aplicar el diseño con tabs
tabs = st.tabs(["Inicio","Datos Crudos", "Gráficas de Sensores", "Gráficas de Análisis"])
with tabs[0]:
    st.title("Bienvenido al Proyecto de Análisis de Datos de Sensores")
    st.write("""
        Este proyecto se centra en el monitoreo y análisis de datos de sensores, específicamente de CO2, temperatura y humedad, para identificar patrones y posibles presencias de personas en un ambiente cerrado.
        
        ### Objetivos del Proyecto:
        - **Monitoreo en tiempo real:** Obtener datos actualizados de los sensores.
        - **Análisis de datos:** Visualizar y analizar los datos de CO2, temperatura y humedad para identificar patrones.
        - **Detección de presencias:** Utilizar técnicas de machine learning, como DBSCAN, para detectar posibles presencias de personas basadas en los datos de CO2.

        ### Secciones de la Aplicación:
        - **Datos Crudos:** Visualización de los datos en su forma original.
        - **Gráficas de Sensores:** Gráficas que muestran las tendencias de CO2, temperatura y humedad.
        - **Gráficas de Análisis:** Análisis avanzado utilizando técnicas de machine learning para detectar patrones y presencias.

        Use el desplegable de arriba para seleccionar el rango de tiempo de los datos que desea analizar.
    """)
with tabs[1]:
    show_raw_data()
with tabs[2]:
    show_sensor_data()
with tabs[3]:
    show_analysis_data()

# Añadir la función para actualizar automáticamente las gráficas cada vez que se cambie la opción del desplegable
st_autorefresh(interval=300000, key=time_range)