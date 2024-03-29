#
# Copyright (c) 2021 Scott Ware
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import base64
import cv2
import grpc
import numpy as np
import json
import os
import paho.mqtt.client as mqtt
import pybase64
import queue
import simplejpeg
import ssl
import threading
import time
from client_utils import prepare_certs, get_model_io_names, get_model_input_shape
from datetime import datetime, timezone
from ovmsclient import make_grpc_client

parser = argparse.ArgumentParser(description='TFS gRPC to MQTT client.')

parser.add_argument('--input',required=False, default='videopub/stream/#', help='Specify the MQTT topic to subscribe')
parser.add_argument('--scale_input', default=False, action='store_true', help='Scale the input to the correct resolution for the model')
parser.add_argument('--header',required=False, default='tfs2mqtt', help='Specify the MQTT topic header')
parser.add_argument('--category',required=False, default='object', help='Specify the default detection category')
parser.add_argument('--threshold', required=False, help='Confidence threshold to include detection', default=0.75, type=float)
parser.add_argument('--workers', required=False, help='Workers for frame processing', default=10, type=int)
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default: localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--mqtt_address',required=False, default='localhost',  help='MQTT broker address. default: localhost')
parser.add_argument('--mqtt_port',required=False, default=1883, help='MQTT port. default: 1883')
parser.add_argument('--mqtt_username',required=False, default='',  help='MQTT username.')
parser.add_argument('--mqtt_password',required=False, default='',  help='MQTT password.')
parser.add_argument('--mqtt_tls', default=False, action='store_true', help='use TLS communication for MQTT')
parser.add_argument('--model_name',required=True, help='Specify the model name')
parser.add_argument('--model_version',required=False, default=0, type=int, help='Specify the model version to use. default: latest')
parser.add_argument('--grpc_tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
parser.add_argument('--perf_stats', default=False, action='store_true', help='Print performance statistics')
parser.add_argument('--debug', default=False, action='store_true', help='Print debug information')
parser.add_argument('--grpc_server_cert', required=False, help='Path to gRPC server certificate')
parser.add_argument('--grpc_client_cert', required=False, help='Path to gRPC client certificate')
parser.add_argument('--grpc_client_key', required=False, help='Path to gRPC client key')
args = vars(parser.parse_args())

address = "{}:{}".format(args.get('grpc_address'),args.get('grpc_port'))

input_topic = args.get('input')
model_name = args.get('model_name')
model_version = args.get('model_version')
header = args.get('header')
category = args.get('category')
threshold = args.get('threshold')
grpc_address = args.get('grpc_address')
grpc_port = args.get('grpc_port')
mqtt_address = args.get('mqtt_address')
mqtt_port = args.get('mqtt_port')
mqtt_username = args.get('mqtt_username')
mqtt_password = args.get('mqtt_password')
num_workers = args.get('workers')
scale_input = args.get('scale_input')
perf_stats = args.get('perf_stats')

mqtt_connected = False

def process_payload(payload):
    try:
        json_payload = json.loads(payload)
    except json.JSONDecodeError:
        return None

    # Check payload
    if "timestamp" in json_payload and \
       "id" in json_payload and \
       "height" in json_payload and \
       "width" in json_payload and \
       "frame" in json_payload:
       return [
              json_payload['timestamp'].replace("+00:00", "Z"),
              json_payload['id'],
              json_payload['height'],
              json_payload['width'],
              json_payload['frame']
              ]

    return None


def process_detections(detections, height, width):
    objects = []

    for i, detection in enumerate(detections):
        _, class_id, confidence, x_min, y_min, x_max, y_max = detection

        if confidence > threshold:
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)
            w = int(x_max - x_min)
            h = int(y_max - y_min)

            if args.get('debug'):
                print("detection", i , detection)

            objects.append({"id":i, "category":category, "class":int(class_id), "confidence":float(confidence), "bounding_box":{"x": x_min, "y": y_min, "width": w, "height": h}})

    return objects


def process_frame(frame, height, width):
    frame = pybase64.b64decode(frame, validate=True)
    frame = np.frombuffer(frame, dtype=np.uint8).tobytes()

    # Scale if requested
    if scale_input:
        if height > input_shape[1] or width > input_shape[2]:
            frame = simplejpeg.decode_jpeg(frame, fastdct=True, fastupsample=True)
            frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
            frame = simplejpeg.encode_jpeg(
                frame,
                quality=85,
                colorspace='BGR',
                colorsubsampling='420',
                fastdct=True,
            )

    inputs = {input_name: frame}

    try:
        results = client.predict(inputs=inputs, model_name=model_name, model_version=model_version)
    except:
        return None

    return results


def frame_worker():
    while True:
        data = q.get()
        payload = process_payload(data)

        if payload is None:
            q.task_done()
            continue

        timestamp = payload[0]
        stream_id = payload[1]
        height = payload[2]
        width = payload[3]
        frame = payload[4]

        if perf_stats:
            start_time = time.perf_counter()

        results = process_frame(frame, height, width)

        if results is None:
            q.task_done()
            continue

        results = results[0].reshape(-1, 7)
        objects = process_detections(results, height, width)

        if perf_stats:
            end_time = time.perf_counter()
            duration = end_time - start_time
            print("ID:", stream_id, '', 'Processing time:', duration)

        if mqtt_connected:
            mqtt_payload = {"timestamp":timestamp,"id":stream_id,"objects":objects}
            mqtt_topic = ''.join([header, "/", "data", "/", "sensor", "/", stream_id, "/", category])
            mqttp.publish(mqtt_topic, json.dumps(mqtt_payload), qos=0, retain=False)

        q.task_done()

#
# Create gRPC client
#

tls_config = None
if args.get('grpc_tls'):
    tls_config = {
        "tls_config": {
            "client_key_path": args.get('grpc_client_key'),
            "client_cert_path": args.get('grpc_client_cert'),
            "server_cert_path": args.get('grpc_server_cert')
        }
    }

client = make_grpc_client(address, tls_config=tls_config)

try:
    input_name, output_name = get_model_io_names(client, model_name, model_version)
    input_shape = get_model_input_shape(client, model_name, model_version)
except Exception as error:
    print(error)
    quit()

#
# Create MQTT clients
#

def on_connect(mqttc, obj, flags, rc):
    print("MQTT connected...")
    mqttc.subscribe(input_topic, 0)

def on_publisher_connect(mqttc, obj, flags, rc):
    global mqtt_connected
    mqtt_connected = True

def on_publisher_disconnect(mqttc):
    global mqtt_connected
    mqtt_connected = False

def on_input_message(mqttc, obj, msg):
    try:
        q.put_nowait(msg.payload.decode("utf-8"))
    except queue.Full:
        return

mqtts = mqtt.Client()
mqtts.message_callback_add(input_topic, on_input_message)
mqtts.on_connect = on_connect
mqtts.username_pw_set(mqtt_username,mqtt_password)

if args.get('mqtt_tls'):
    mqtts.tls_set(cert_reqs=ssl.CERT_NONE)
    mqtts.tls_insecure_set(True)

mqtts.connect(mqtt_address, int(mqtt_port), 60)

mqttp = mqtt.Client()
mqttp.on_connect = on_publisher_connect
mqttp.on_disconnect = on_publisher_disconnect
mqttp.username_pw_set(mqtt_username,mqtt_password)

if args.get('mqtt_tls'):
    mqttp.tls_set(cert_reqs=ssl.CERT_NONE)
    mqttp.tls_insecure_set(True)

mqttp.connect(mqtt_address, int(mqtt_port), 60)

# Limit OpenCV thread pool
cv2.setNumThreads(num_workers)

# Initialise worker queue
q = queue.Queue(maxsize=num_workers * 2)

print('Start processing frames...')

#Starting workers
for i in range(num_workers):
    worker = threading.Thread(target=frame_worker, daemon=True, args=())
    worker.start()

mqttp.loop_start()
mqtts.loop_forever()
