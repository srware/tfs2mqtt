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
import json
import os
import paho.mqtt.publish as publish
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import prepare_certs
from datetime import datetime, timezone

parser = argparse.ArgumentParser(description='TFS gRPC to MQTT client.')

parser.add_argument('--input',required=False, default='/dev/video0', help='Specify the video input')
parser.add_argument('--header',required=False, default='tfs2mqtt', help='Specify the MQTT topic header')
parser.add_argument('--id',required=False, default='camera1', help='Unique identifier for MQTT topic')
parser.add_argument('--category',required=False, default='object', help='Specify the default detection category')
parser.add_argument('--width', required=False, help='How the input image width should be resized in pixels', default=600, type=int)
parser.add_argument('--height', required=False, help='How the input image width should be resized in pixels', default=600, type=int)
parser.add_argument('--confidence', required=False, help='Confidence threshold to include detection', default=0.75, type=float)
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--mqtt_address',required=False, default='localhost',  help='MQTT broker address. default:localhost')
parser.add_argument('--mqtt_port',required=False, default=1883, help='MQTT port. default: 1883')
parser.add_argument('--model_name',required=True, help='Specify the model name')
parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
parser.add_argument('--perf_stats', default=False, action='store_true', help='Print performance statistics')
parser.add_argument('--debug', default=False, action='store_true', help='Print debug information')
parser.add_argument('--server_cert', required=False, help='Path to server certificate')
parser.add_argument('--client_cert', required=False, help='Path to client certificate')
parser.add_argument('--client_key', required=False, help='Path to client key')
args = vars(parser.parse_args())

address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

mqtt_topic = ''.join([args['header'], "/", "data", "/", "sensor", "/", args['id'], "/", args['category']])

channel = None
if args.get('tls'):
    server_ca_cert, client_key, client_cert = prepare_certs(server_cert=args['server_cert'],
                                                            client_key=args['client_key'],
                                                            client_ca=args['client_cert'])
    creds = grpc.ssl_channel_credentials(root_certificates=server_ca_cert,
                                         private_key=client_key, certificate_chain=client_cert)
    channel = grpc.secure_channel(address, creds)
else:
    channel = grpc.insecure_channel(address)

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

vcap = cv2.VideoCapture()
vcap.set(cv2.CAP_PROP_FPS, 15)

if(vcap.open(args['input']) == False):
    print('Failed to open camera stream... quitting!')
    quit()

print('Start processing frames...')

while(1):
    ret, img = vcap.read()

    if(ret == False):
        print('Failed to get frame from camera...')
        continue;

    # Get original image shape
    orig_height, orig_width, orig_channels = img.shape

    timestamp_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    img = cv2.resize(img, (args['width'], args['height']))
    img = img.transpose(2,0,1).reshape(1,3,args['height'],args['width'])

    imgs = np.zeros((0,3,args['height'],args['width']), np.dtype('<f'))
    imgs = np.append(imgs, img, axis=0)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = args['model_name']
    request.inputs["data"].CopyFrom(make_tensor_proto(imgs, shape=(imgs.shape)))

    if args.get('perf_stats'):
        start_time = datetime.now()

    result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs

    if args.get('perf_stats'):
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print('Processing time: {:.2f} ms; speed {:.2f} fps'.format(round(duration, 2), round(1000 / duration, 2)))

    result = make_ndarray(result.outputs["detection_out"])
    detections = result.reshape(-1, 7)

    objects = []

    for i, detection in enumerate(detections):
        _, class_id, confidence, x_min, y_min, x_max, y_max = detection

        if confidence > args['confidence']:
            x_min = int(x_min * orig_width)
            y_min = int(y_min * orig_height)
            x_max = int(x_max * orig_width)
            y_max = int(y_max * orig_height)
            w = int(x_max - x_min)
            h = int(y_max - y_min)

            if args.get('debug'):
                print("detection", i , detection)

            objects.append({"id":i, "category":args['category'],"confidence":float(confidence), "bounding_box":{"x": x_min, "y": y_min, "width": w, "height": h}})

    mqtt_payload ={"timestamp":timestamp_str,"id":args['id'], "objects":objects}
    publish.single(mqtt_topic, json.dumps(mqtt_payload), hostname=args['mqtt_address'], port=args['mqtt_port'], auth=None)
