import streamlit as st
import requests
import datetime
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Petición a la API REST para obtener datos del último mes
def fetch_data_from_api():
    mes_pasado = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp()) * 1000
    url = "https://sensecap.seeed.cc/openapi/list_telemetry_data"
    auth = ('93I2S5UCP1ISEF4F', '6552EBDADED14014B18359DB4C3B6D4B3984D0781C2545B6A33727A4BBA1E46E')
    
    # Parámetros de la solicitud GET
    params = {
        'device_eui': '2CF7F1C044300627',
        'time_start': mes_pasado
    }

    response = requests.get(url, params=params, auth=auth)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error al realizar la solicitud. Código de estado: {response.status_code}")
        return None

# Procesar los datos recibidos de la API y almacenarlos en un DataFrame
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

# Detectar anomalías utilizando DBSCAN
def detect_anomalies(df, eps=5, min_samples=20):
    df = df.dropna(subset=['CO2'])  # Eliminar filas con valores NaN en CO2
    X = df[['CO2']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['anomaly'] = dbscan.fit_predict(X)
    return df

# Clasificar día y noche utilizando regresión logística
def classify_day_night(df):
    df = df.dropna(subset=['temperature'])
    
    # Crear una columna binaria para día (1) y noche (0)
    df['is_day'] = df['timestamp'].apply(lambda x: 1 if 7 <= x.hour <= 21 else 0)
    
    X = df[['temperature']]
    y = df['is_day']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    accuracy = model.score(X_test, y_test)
    st.write(f"Precisión del modelo: {accuracy:.2f}")
    
    df['day_night_prediction'] = model.predict(scaler.transform(df[['temperature']]))
    
    return df

# Configuración de la página de Streamlit
st.title("Monitor de CO2 en Tiempo Real")

# Espacio reservado para la gráfica
chart_placeholder = st.empty()
day_night_chart_placeholder = st.empty()

# Función para crear y mostrar la gráfica de CO2
def plot_data(df):
    fig, ax = plt.subplots()
    ax.plot(df['timestamp'], df['CO2'], label='CO2')
    ax.scatter(df[df['anomaly'] == -1]['timestamp'], df[df['anomaly'] == -1]['CO2'], color='red', label='Anomalías')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('CO2')
    ax.legend()

    # Ajustar las etiquetas de las fechas
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Rotar y ajustar el tamaño de las etiquetas

    st.pyplot(fig)

# Función para crear y mostrar la gráfica de día y noche
def plot_day_night(df):
    fig, ax = plt.subplots()
    ax.plot(df['timestamp'], df['day_night_prediction'], label='Predicción Día/Noche', color='purple')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Día (1) / Noche (0)')
    ax.legend()

    # Ajustar las etiquetas de las fechas
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Rotar y ajustar el tamaño de las etiquetas

    st.pyplot(fig)

# Obtener y procesar los datos de la API
api_data = fetch_data_from_api()
df = process_api_data(api_data)

# Detectar anomalías
df = detect_anomalies(df)

# Clasificar día y noche
df = classify_day_night(df)

# Mostrar la gráfica inicial
if not df.empty:
    with chart_placeholder:
        plot_data(df)
    with day_night_chart_placeholder:
        plot_day_night(df)
else:
    st.write("No hay datos disponibles para mostrar.")

# Mantener la aplicación actualizada con los nuevos datos cada minuto
# while True:
#     api_data = fetch_data_from_api()
#     df = process_api_data(api_data)
#     if not df.empty:
#         df = detect_anomalies(df)
#         with chart_placeholder:
#             plot_data(df)
#     time.sleep(60)  # Esperar 60 segundos antes de actualizar