import socket
import time
from _thread import *
import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

# Assuming you have YOLOv5 model setup as in your previous code
weights = '/Users/eunbin/Desktop/GDSC/models/new.pt'  # model path
data = '/Users/eunbin/Desktop/GDSC/yolov5/custom.yaml'  # dataset.yaml path
device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz = (640, 640)  # inference size (height, width)

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=data)
stride = model.stride
names = model.names

def load_byte_image(byte_data):
    np_arr = np.frombuffer(byte_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def run_inference(byte_data):
    # Convert byte data to image
    img0 = load_byte_image(byte_data)
    img = torch.from_numpy(img0).to(device)
    img = img.float()  # uint8 to fp32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model(img, augment=False, visualize=False)

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img size to img0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                print(label, xyxy)  # Print label and bounding box

def threaded(client_socket, addr):
    print('>> Connected by :', addr[0], ':', addr[1])
    start = time.time()

    image_data_bytes = b""
    while True:
        try:
            data = client_socket.recv(4096)
            if not data:
                break
            image_data_bytes += data
        except ConnectionResetError as e:
            print('>> Disconnected (ConnectionError) ' + addr[0], ':', addr[1])
            break

    # Run YOLOv5 inference on the received image data
    run_inference(image_data_bytes)

    end = time.time()
    print(f"{end - start:.5f} sec")
    client_socket.close()

if __name__ == "__main__":
    host = '127.0.0.1'
    port = 9999

    print('>> Server Start with ip :', host)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"ML 소켓 서버가 {host}:{port}에서 시작되었습니다.")
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Client connect 연결: {addr}")
        start_new_thread(threaded, (client_socket, addr))