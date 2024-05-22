import streamlit as st
import requests
import datetime
import pandas as pd
import time

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
        return pd.DataFrame(columns=['timestamp', 'CO2'])

    canal_identificacion = api_data['data']['list'][0]
    valores_medidos = api_data['data']['list'][1]

    data = []
    for tipo_med, _ in canal_identificacion:
        if _ == "4100":  # Identificar solo los datos de CO2
            for sublist in valores_medidos:
                for valor, timestamp in sublist:
                    fecha = pd.to_datetime(timestamp)
                    valor = float(valor) if isinstance(valor, (int, float, str)) else None
                    data.append({'timestamp': fecha, 'CO2': valor})

    return pd.DataFrame(data)

# Configuración de la página de Streamlit
st.title("Monitor de CO2 en Tiempo Real")

# Espacio reservado para la gráfica
chart_placeholder = st.empty()

# Obtener y procesar los datos de la API
api_data = fetch_data_from_api()
df = process_api_data(api_data)

# Mostrar y actualizar la gráfica
if not df.empty:
    with chart_placeholder:
        st.line_chart(df.set_index('timestamp')['CO2'])
else:
    st.write("No hay datos disponibles para mostrar.")

# Mantener la aplicación actualizada con los nuevos datos cada minuto
while True:
    api_data = fetch_data_from_api()
    df = process_api_data(api_data)
    if not df.empty:
        with chart_placeholder:
            st.line_chart(df.set_index('timestamp')['CO2'])
    time.sleep(60)  # Esperar 60 segundos antes de actualizar