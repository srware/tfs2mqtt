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
import ssl
from client_utils import prepare_certs, get_model_io_names
from datetime import datetime, timezone
from ovmsclient import make_grpc_client

parser = argparse.ArgumentParser(description='TFS gRPC to MQTT client.')

parser.add_argument('--input',required=False, default='/dev/video0', help='Specify the video input')
parser.add_argument('--loop', default=False, action='store_true', help='Loop input video')
parser.add_argument('--header',required=False, default='tfs2mqtt', help='Specify the MQTT topic header')
parser.add_argument('--id',required=False, default='camera1', help='Unique identifier for MQTT topic')
parser.add_argument('--category',required=False, default='object', help='Specify the default detection category')
parser.add_argument('--width', required=False, help='Specify desired input image width', type=int)
parser.add_argument('--height', required=False, help='Specify desired input image height', type=int)
parser.add_argument('--threshold', required=False, help='Confidence threshold to include detection', default=0.75, type=float)
parser.add_argument('--threads', required=False, help='Limit CPU usage for camera processing', default=10, type=int)
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

input = args.get('input')
model_name = args.get('model_name')
model_version = args.get('model_version')
instance_id = args.get('id')
header = args.get('header')
category = args.get('category')
threshold = args.get('threshold')
grpc_address = args.get('grpc_address')
grpc_port = args.get('grpc_port')
mqtt_address = args.get('mqtt_address')
mqtt_port = args.get('mqtt_port')
mqtt_username = args.get('mqtt_username')
mqtt_password = args.get('mqtt_password')

command_topic = ''.join([header, "/", "cmd", "/", "sensor", "/", "cam", "/", instance_id])
image_topic = ''.join([header, "/", "image", "/", "sensor", "/", "cam", "/", instance_id])

# Camera frame & timestamp
curr_frame = None
curr_timestamp = None
mode = None

def process_frame(timestamp, id, height, width, frame):
    if args.get('perf_stats'):
        start_time = datetime.now()

    input_name, output_name = get_model_io_names(client, model_name, model_version)
    inputs = {input_name: frame.tobytes() }
    results = client.predict(inputs=inputs, model_name=model_name)

    if args.get('perf_stats'):
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print('Processing time: {:.2f} ms; speed {:.2f} fps'.format(round(duration, 2), round(1000 / duration, 2)))

    detections = results[0].reshape(-1, 7)

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

    mqtt_payload = {"timestamp":timestamp,"id":id,"objects":objects}
    mqtt_topic = ''.join([header, "/", "data", "/", "sensor", "/", category, "/", id])
    mqttc.publish(mqtt_topic, json.dumps(mqtt_payload))

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

#
# Create MQTT client
#

def on_connect(mqttc, obj, flags, rc):
    print("MQTT connected...")

def on_message(mqttc, obj, msg):
    topic = msg.topic
    payload = msg.payload.decode("utf-8")

    if payload == 'getimage':
        if curr_frame is None:
            return

        jpeg = base64.b64encode(curr_frame).decode('utf-8')
        image_payload = {'timestamp':curr_timestamp.isoformat(timespec='milliseconds').replace("+00:00", "Z"), 'id':instance_id, 'image':jpeg}
        mqttc.publish(image_topic, json.dumps(image_payload))

    else:
        try:
            json_payload = json.loads(payload)
        except JSONDecodeError:
            return

        # Check payload
        if "timestamp" in json_payload and "id" in json_payload and "height" in json_payload and "width" in json_payload and "frame" in json_payload:
            img = base64.b64decode(json_payload['frame'])
            process_frame(json_payload['timestamp'], json_payload['id'], json_payload['height'], json_payload['width'], np.frombuffer(img, dtype=np.uint8))

mqttc = mqtt.Client()
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.username_pw_set(mqtt_username,mqtt_password)

if args.get('mqtt_tls'):
    mqttc.tls_set(cert_reqs=ssl.CERT_NONE)
    mqttc.tls_insecure_set(True)

mqttc.connect(mqtt_address, int(mqtt_port), 60)

# Limit OpenCV thread pool
cv2.setNumThreads(args['threads'])

# Test for MQTT stream
if input.startswith('mqtt:'):
    mqttc.subscribe(input.split(':')[1], 0)
    mode = 'mqtt'
else:
    vcap = cv2.VideoCapture()
    status = vcap.open(args['input'])

    # Test for camera stream
    if not status:
        print('Failed to open video stream... quitting!')
        quit()

    mode = 'video'
    mqttc.subscribe(command_topic, 0)

print('Start processing frames...')

if mode == 'video':
    mqttc.loop_start()
    while(1):
        status, img = vcap.read()

        if not status:
            # Loop video if applicable
            if not args.get('loop'):
                print('No more frames available... Quitting!')
                quit()

            vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status, img = vcap.read()

            if not status:
                quit()

        curr_timestamp = datetime.now(timezone.utc)
        timestamp_str = curr_timestamp.isoformat(timespec='milliseconds').replace("+00:00", "Z")

        # Scale if requested
        if args.get('width') and args.get('height'):
            img = cv2.resize(img, (args.get('width'), args.get('height')))

        # Get original image shape
        ret, curr_frame = cv2.imencode(".jpg", img)
        height, width, channels = img.shape

        process_frame(timestamp_str, instance_id, height, width, img)
else:
    mqttc.loop_forever()
