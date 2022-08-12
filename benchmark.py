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
import cv2
import grpc
import numpy as np
import os
import queue
import simplejpeg
import ssl
import threading
import time
import uuid
import paho.mqtt.client as mqtt
from client_utils import prepare_certs, get_model_io_names, get_model_input_shape
from datetime import datetime, timezone
from ovmsclient import make_grpc_client
from time import sleep

parser = argparse.ArgumentParser(description='TFS gRPC benchmark utility.')

parser.add_argument('--input', required=True, help='Input image')
parser.add_argument('--id', required=False, help='Unique identifier for benchmark')
parser.add_argument('--workers', required=False, help='Workers for frame processing', default=10, type=int)
parser.add_argument('--scale_input', default=False, action='store_true', help='Scale the input to the correct resolution for the model')
parser.add_argument('--infer_only', default=False, action='store_true', help='Pre-process input and benchmark inference only')
parser.add_argument('--binary_input', default=False, action='store_true', help='Use binary inputs')
parser.add_argument('--mqtt_address',required=False, default='localhost',  help='MQTT broker address. default: localhost')
parser.add_argument('--mqtt_port',required=False, default=1883, help='MQTT port. default: 1883')
parser.add_argument('--mqtt_username',required=False, default='',  help='MQTT username.')
parser.add_argument('--mqtt_password',required=False, default='',  help='MQTT password.')
parser.add_argument('--mqtt_tls', default=False, action='store_true', help='use TLS communication for MQTT')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default: localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name',required=True, help='Specify the model name')
parser.add_argument('--model_version',required=False, default=0, type=int, help='Specify the model version to use. default: latest')
parser.add_argument('--grpc_tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
parser.add_argument('--grpc_server_cert', required=False, help='Path to gRPC server certificate')
parser.add_argument('--grpc_client_cert', required=False, help='Path to gRPC client certificate')
parser.add_argument('--grpc_client_key', required=False, help='Path to gRPC client key')
parser.add_argument('--perf_stats', default=False, action='store_true', help='Print detailed performance statistics')
args = vars(parser.parse_args())

address = "{}:{}".format(args.get('grpc_address'),args.get('grpc_port'))

image_file = args.get('input')
instance_id = args.get('id')
model_name = args.get('model_name')
model_version = args.get('model_version')
mqtt_address = args.get('mqtt_address')
mqtt_port = args.get('mqtt_port')
mqtt_username = args.get('mqtt_username')
mqtt_password = args.get('mqtt_password')
grpc_address = args.get('grpc_address')
grpc_port = args.get('grpc_port')
num_workers = args.get('workers')
scale_input = args.get('scale_input')
infer_only = args.get('infer_only')
binary_input = args.get('binary_input')
perf_stats = args.get('perf_stats')

# Performance counters
count = 0
total_latency = 0

if not instance_id:
    instance_id = str(uuid.uuid4())

mqtt_topic = "tfs2mqtt/benchmark/" + instance_id

def load_image(path):
    with open(path, 'rb') as f:
        return f.read()

def prepare_input(input_img):
    scaling_required = False
    decode_required = False
    
    # Decode JPEG header
    height, width, colorspace, subsampling = simplejpeg.decode_jpeg_header(input_img)

    # Check if scaling is required
    if scale_input or not binary_input:
        if width > model_width or height > model_height:
            scaling_required = True

    # Check if decoding is required
    if scaling_required or not binary_input:
        decode_required = True

    if decode_required:
        frame = simplejpeg.decode_jpeg(input_img, fastdct=True, fastupsample=True)

    if scaling_required:
        frame = cv2.resize(frame, (model_width, model_height))    

    if decode_required and binary_input:
        input_img = simplejpeg.encode_jpeg(
            frame,
            quality=85,
            colorspace='BGR',
            colorsubsampling='420',
            fastdct=True,
        )

    if not binary_input:
        frame = frame.transpose(2,0,1).reshape(1,input_shape[1],input_shape[2],input_shape[3])
        input_img = np.zeros((0,input_shape[1],input_shape[2],input_shape[3]), np.float32)
        input_img = np.append(input_img, frame, axis=0)
 
    return {input_name: input_img }

def benchmark_worker():
    # Prepare inputs
    if infer_only:
        inputs = prepare_input(img)

    while True:
        start_time = time.perf_counter()

        if not infer_only:
            inputs = prepare_input(img)

        try:
            results = client.predict(inputs=inputs, model_name=model_name)
        except:
            continue

        end_time = time.perf_counter()
        duration = (end_time - start_time)

        mqttp.publish(mqtt_topic, str(duration), qos=0, retain=False)

        if perf_stats:
            print('Processing time:', round(duration, 3))


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

model_height = input_shape[1]
model_width = input_shape[2]

#
# Create MQTT clients
#

def on_connect(mqttc, obj, flags, rc):
    print("MQTT connected...")
    mqttc.subscribe(mqtt_topic, 0)

def on_message(mqttc, obj, msg):
    global count, total_latency

    count += 1
    total_latency += float(msg.payload.decode("utf-8"))

mqtts = mqtt.Client()
mqtts.message_callback_add(mqtt_topic, on_message)
mqtts.on_connect = on_connect
mqtts.username_pw_set(mqtt_username,mqtt_password)

if args.get('mqtt_tls'):
    mqtts.tls_set(cert_reqs=ssl.CERT_NONE)
    mqtts.tls_insecure_set(True)

mqtts.connect(mqtt_address, int(mqtt_port), 60)
mqtts.loop_start()

mqttp = mqtt.Client()
mqttp.username_pw_set(mqtt_username,mqtt_password)

if args.get('mqtt_tls'):
    mqttp.tls_set(cert_reqs=ssl.CERT_NONE)
    mqttp.tls_insecure_set(True)

mqttp.connect(mqtt_address, int(mqtt_port), 60)
mqttp.loop_start()

# Read image
img = load_image(image_file)

# Initialise worker queue
q = queue.Queue(maxsize=0)

print('Start benchmarking...')

# Starting workers
for i in range(num_workers):
    threading.Thread(target=benchmark_worker, args=(), daemon=True).start()

while True:
    count = 0
    total_latency = 0

    sleep(1)

    average_latency = 0

    if count > 0:
        average_latency = total_latency / count

    print('Frames Per Second:', count, " ", 'Average Latency:', round(average_latency, 3))
