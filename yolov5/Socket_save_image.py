import socket
import time
from _thread import *
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

file_number = 1

# Assuming YOLOv5 model setup
weights = '/home/dnflrha12/PathPal-ML/yolov5/models/new.pt'  # model path
data = '/home/dnflrha12/PathPal-ML/yolov5/custom.yaml'  # dataset.yaml path
device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
imgsz = (640, 640)  # inference size (height, width)

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=data)
stride = model.stride
names = model.names

def resize_image(img, img_size=640):
    # 이미지의 원본 크기와 대상 크기를 계산합니다.
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    
    # 새로운 이미지 크기를 계산합니다.
    nh, nw = int(round(h * scale)), int(round(w * scale))
    
    # 이미지를 재조정합니다.
    resized_img = cv2.resize(img, (nw, nh))

    # 패딩을 추가하여 이미지를 정사각형으로 만듭니다.
    top_pad = (img_size - nh) // 2
    bottom_pad = img_size - nh - top_pad
    left_pad = (img_size - nw) // 2
    right_pad = img_size - nw - left_pad
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    return padded_img
def run_inference(image_path, img_size=640):
    img0 = cv2.imread(image_path)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    
    # 이미지 재조정 및 패딩 추가
    resized_img = resize_image(img0, img_size)
    
    # NumPy 배열을 PyTorch 텐서로 변환
    img = torch.from_numpy(resized_img).to(device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    # 모델을 통해 추론 수행
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    # 결과 처리
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], resized_img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                print(label, xyxy)


def save_image_from_bytes(data, filename):
    with open(filename, 'wb') as f:
        f.write(data)

def threaded(client_socket, addr):
    print('>> Connected by :', addr[0], ':', addr[1])
    image_data_bytes = b''
    while True:
        try:
            data = client_socket.recv(1024)
            if not data:
                break
            image_data_bytes += data
        except ConnectionResetError as e:
            print("Connection Reset Error")
            break

    global file_number
    file_name = f'image_{file_number}.jpg'
    file_number += 1
    save_image_from_bytes(image_data_bytes, file_name)

    # Run inference on the saved image
    run_inference(file_name)

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
