import cv2
import json
import socket
import numpy as np
import tensorflow as tf
import time

ALERT_LABELS = [
    'flatness_D', 'flatness_E', 'paved_state_broken', 'block_kind_bad', 'outcurb_rectengle_broken',
    'sidegap_out', 'steepramp', 'brailleblock_dot_broken', 'brailleblock_line_broken', 'planecrosswalk_broken',
    'pole', 'bollard', 'barricade', 'stair'
]

def load_model(model_path='model.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def resize_and_pad_image(image, img_size=640):
    shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    ratio = img_size / tf.reduce_max(shape)
    new_shape = tf.round(shape * ratio)
    image = tf.image.resize(image, tf.cast(new_shape, dtype=tf.int32))
    pad_width, pad_height = img_size - new_shape[1], img_size - new_shape[0]
    image = tf.image.pad_to_bounding_box(image, 0, 0, img_size, img_size)
    return image

def preprocess_image(image_path, img_size=640):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32) / 255.0
    image = resize_and_pad_image(image, img_size)
    image = tf.expand_dims(image, axis=0)
    return image

def run_inference(image_path, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = preprocess_image(image_path)
    img = np.array(img, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box 좌표
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    return process_detection(boxes, classes, scores, img.shape)

def process_detection(boxes, classes, scores, image_shape):
    processed_results = []
    for box, score, cls_id in zip(boxes, scores, classes):
        if score >= 0.5:
            y_min, x_min, y_max, x_max = box
            label = ALERT_LABELS[int(cls_id)]
            alert = label in ALERT_LABELS
            processed_result = {
                "name": label,
                "confidence": float(score),
                "left_x": float(x_min),
                "right_x": float(x_max),
                "down_y": float(y_max),
                "up_y": float(y_min),
                "alert": alert
            }
            processed_results.append(processed_result)

    json_output = json.dumps(processed_results, indent=4)
    return json_output
