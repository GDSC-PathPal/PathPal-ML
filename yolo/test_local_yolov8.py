import socket
import time
from _thread import *
import cv2
import json
import torch
from ultralytics import YOLO 
import os
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

file_number = 1
# image_path = '/home/dnflrha12/PathPal-ML/yolov5/example2.jpg' 
weights = '/home/dnflrha12/PathPal-ML/yolov5/yolov8_epoch50_add_label.pt' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ALERT_LABELS = [
    'flatness_D', 'flatness_E', 'paved_state_broken', 'block_kind_bad', 'outcurb_rectengle','outcurb_slide', 'outcurb_rectengle_broken',
    'sidegap_out', 'steepramp', 'brailleblock_dot_broken', 'brailleblock_line_broken', 'planecrosswalk_broken',
    'pole', 'bollard', 'barricade'
]

def resize_image(img, img_size=640):
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized_img = cv2.resize(img, (nw, nh))
    top_pad = (img_size - nh) // 2
    bottom_pad = img_size - nh - top_pad
    left_pad = (img_size - nw) // 2
    right_pad = img_size - nw - left_pad
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return padded_img

def process_detection(results):
    processed_results = []
    for r in results:
        total_width = r.orig_shape[1]
        print("이미지 너비 : ",total_width)
        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()[:4]
            conf = r.boxes.conf[i].item()
            cls_id = int(r.boxes.cls[i].item())
            label = r.names[cls_id]
            alert = label in ALERT_LABELS
            
            result = {
                "name": label,
                "confidence": round(conf, 4),
                "left_x": round(x1 / total_width, 2),
                "right_x": round(x2 / total_width, 2),
                "alert": alert
            }
            processed_results.append(result)
    
    json_output = json.dumps(processed_results, indent=4)
    # print(json_output)
    return json_output

# 추론 실행 함수
def run_inference(image_path, img_size=640):
    img0 = cv2.imread(image_path)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    resized_img = resize_image(img0, img_size)
    print("리사이징된 이미지 크기:", resized_img.shape)  # 여기에 크기 출력 코드 추가

    img = torch.from_numpy(resized_img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # 모델 추론
    model = YOLO(weights) 
    results = model(img)
    
    # 결과 후처리 및 출력
    results = process_detection(results)
    return results

image_path = '/home/dnflrha12/PathPal-ML/yolov5/example.jpg'
result = run_inference(image_path)
print(result)