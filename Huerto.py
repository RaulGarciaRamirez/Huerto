import streamlit as st
import paho.mqtt.client as mqtt
import json
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# Datos de conexión
username = 'org-434181208382464'
password = '6552EBDADED14014B18359DB4C3B6D4B3984D0781C2545B6A33727A4BBA1E46E'
host = 'sensecap-openstream.seeed.cc'
port = 1883
topic = "/device_sensor_data/434181208382464/2CF7F1C04430065C/+/+/+"
client_id = f'org-434181208382464-{random.randint(0, 100)}'

# Inicializar una lista para almacenar los datos de CO2
co2_data = []

# Inicializar el dataframe
df = pd.DataFrame(columns=['timestamp', 'CO2'])

# Callback para cuando nos conectamos al broker
def on_connect(client, userdata, flags, rc):
    print(f"Conectado con código de resultado {rc}")
    client.subscribe(topic)

# Callback para cuando recibimos un mensaje
def on_message(client, userdata, msg):
    global df
    payload = json.loads(msg.payload)
    topic_parts = msg.topic.split('/')
    measurement_code = int(topic_parts[6])

    # Mapeo de códigos de medida
    if measurement_code == 4100:  # CO2
        value = payload['value']
        timestamp = pd.to_datetime(payload['timestamp'], unit='ms')
        new_data = pd.DataFrame({'timestamp': [timestamp], 'CO2': [value]})
        df = pd.concat([df, new_data], ignore_index=True)
        df = df.tail(50)  # Mantener solo los últimos 50 registros

# Crear instancia del cliente MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,client_id=client_id)
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message

# Conectar al broker MQTT
client.connect(host, port, 60)

# Ejecutar el loop del cliente MQTT en un hilo separado
client.loop_start()

# Configuración de la página de Streamlit
st.title("Monitor de CO2 en Tiempo Real")

# Función para actualizar la gráfica
def plot_co2_data():
    st.line_chart(df.set_index('timestamp')['CO2'])

# Espacio reservado para la gráfica
chart_placeholder = st.empty()

# Bucle principal de la aplicación de Streamlit
while True:
    if not df.empty:
        with chart_placeholder:
            st.line_chart(df.set_index('timestamp')['CO2'])
    time.sleep(60)  # Esperar 5 segundos antes de actualizar

# Parar el loop del cliente MQTT al finalizar
client.loop_stop()
client.disconnect()
