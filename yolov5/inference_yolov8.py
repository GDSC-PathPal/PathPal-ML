import cv2
import json
import socket
import torch
import time
from ultralytics import YOLO 

file_number = 1
weights = '/home/dnflrha12/PathPal-ML/yolov5/yolov8s_epoch80_stairs_and_greenlight.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ALERT_LABELS = [
    'flatness_D', 'flatness_E', 'paved_state_broken', 'block_kind_bad', 'outcurb_rectengle_broken',
    'sidegap_out', 'steepramp', 'brailleblock_dot_broken', 'brailleblock_line_broken', 'planecrosswalk_broken',
    'pole', 'bollard', 'barricade', 'stair'
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
        total_height = r.orig_shape[0]  # 이미지 높이
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
                "down_y" : round(y2 / total_height, 2),  
                "up_y" : round(y1 / total_height, 2), 
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
    img = torch.from_numpy(resized_img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # 모델 추론
    model = YOLO(weights) 
    results = model(img)
    
    # 결과 후처리 및 출력
    results = process_detection(results)
    return results

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
    print('image 전송됨')

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
