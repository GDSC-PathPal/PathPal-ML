import torch
import torch.onnx
from ultralytics import YOLO 

# 모델 로드
weights = '/home/dnflrha12/PathPal-ML/yolov5/yolov8s_epoch80_stairs_and_greenlight.pt'

model = YOLO(weights) 
model.eval()

# 더미 입력 데이터 생성
example_input = torch.randn(1, 3, 640, 640)

# 모델을 ONNX로 내보내기
torch.onnx.export(model, example_input, 'model.onnx', export_params=True)