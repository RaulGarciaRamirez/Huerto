import paho.mqtt.client as mqtt
import json
import random
import time

# Datos de conexión
username = 'org-434181208382464'
password = '6552EBDADED14014B18359DB4C3B6D4B3984D0781C2545B6A33727A4BBA1E46E'
host = 'sensecap-openstream.seeed.cc'
port = 1883
topic = "/device_sensor_data/434181208382464/2CF7F1C04430065C/+/+/+"
client_id = f'org-434181208382464-{random.randint(0, 100)}'

# Callback para cuando nos conectamos al broker
def on_connect(client, userdata, flags, rc):
    print(f"Conectado con código de resultado {rc}")
    client.subscribe(topic)

# Callback para cuando recibimos un mensaje
def on_message(client, userdata, msg):
    print(f"Mensaje recibido en topic {msg.topic}")
    payload = json.loads(msg.payload)
    print(f"Datos: {payload}")

    topic_parts = msg.topic.split('/')
    device_eui = topic_parts[2]
    measurement_code = int(topic_parts[6])

    # Mapeo de códigos de medida
    measurement_map = {
        4097: 'Temperatura',
        4098: 'Humedad',
        4100: 'CO2'
    }

    measurement_type = measurement_map.get(measurement_code, 'Desconocido')
    value = payload['value']

    print(f"Dispositivo: {device_eui}, Tipo de medida: {measurement_type}, Valor: {value}")

    # Detección de persona basada en CO2
    global last_detection_time
    current_time = time.time()

    if measurement_type == 'CO2' and value > CO2_THRESHOLD:
        if current_time - last_detection_time > DETECTION_INTERVAL:
            print("Posible presencia de persona detectada en el huerto")
            last_detection_time = current_time

# Crear instancia del cliente MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,client_id=client_id)  # No especificamos la versión del protocolo aquí
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message

# Conectar al broker MQTT
client.connect(host, port, 60)

# Variables para la detección de persona
CO2_THRESHOLD = 500  # Ejemplo de umbral para CO2
last_detection_time = 0
DETECTION_INTERVAL = 300  # 5 minutos

# Iniciar el loop para recibir mensajes
client.loop_forever()
