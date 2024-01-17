import socket
import time
from _thread import *
import cv2
import json
import torch
import os
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


file_number = 1
weights = '/home/dnflrha12/PathPal-ML/yolov5/models/best.pt'  # model path
data = '/home/dnflrha12/PathPal-ML/yolov5/custom.yaml'  # dataset.yaml path
device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz = (640, 640)  # inference size (height, width)
# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=data)
stride = model.stride
names = model.names
ALERT_LABELS = ['label1', 'label2']

def resize_image(img, img_size=640):
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    
    # new image size
    nh, nw = int(round(h * scale)), int(round(w * scale))
    
    # resize
    resized_img = cv2.resize(img, (nw, nh))

    # padding
    top_pad = (img_size - nh) // 2
    bottom_pad = img_size - nh - top_pad
    left_pad = (img_size - nw) // 2
    right_pad = img_size - nw - left_pad
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    return padded_img


def run_inference(image_path, img_size=640):
    img0 = cv2.imread(image_path)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    resized_img = resize_image(img0, img_size)

    img = torch.from_numpy(resized_img).to(device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    total_width = img0.shape[1] # 640 fix
    print(total_width)
    results = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], resized_img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                left_x, top_y, right_x, bottom_y = xyxy
                label = names[int(cls)]
                alert = label in ALERT_LABELS
                
                result = {
                    "name": label,
                    "confidence": round(float(conf),4),
                    "left_x": round(float(left_x) / total_width * 100, 2),
                    "right_x": round(float(right_x) / total_width * 100, 2),
                    "alert": alert
                }
                results.append(result)

    # Output the results in JSON format
    json_output = json.dumps(results, indent=4)
    print(json_output)
    return json_output


def save_image_from_bytes(data, filename):
    with open(filename, 'wb') as f:
        f.write(data)

def execute(client_socket):
    global file_number

    while True:
        file_size = client_socket.recv(4)
        if not file_size:
            raise Exception('받은 데이터가 정상이 아님') 
        file_size_int = int.from_bytes(file_size, byteorder='big')
        if file_size_int != 0 :
            break

    start = time.time()
    print('File Size : ' + str(file_size_int))

    client_socket.sendall(file_size)

    image_data_bytes = b''
    while True:
        try:
            data = client_socket.recv(2048)
            image_data_bytes += data
            if image_data_bytes.__len__() == file_size_int:
                break
        except Exception as e:
            print("error: " + e)
            break

    file_name = "saved_images/" + str(file_number) + '.jpg'
    file_number += 1
    save_image_from_bytes(image_data_bytes, file_name)
    
    result = run_inference(file_name)
    # result = run_inference(image_data_bytes)
    client_socket.sendall(result.encode())

    end = time.time()
    print(f"{end - start:.5f} sec")


if __name__ == "__main__":
    host = '127.0.0.1'
    port = 9999

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"ML 소켓 서버가 {host}:{port}에서 시작되었습니다.")

        while True:
            print("Listen")
            client_socket, addr = server_socket.accept()  # block
            print(f"Client connect 연결: {addr}")

            try:
                while True:
                    execute(client_socket)
            except Exception as e:
                print('err: ', e)
            finally:
                client_socket.close()
            