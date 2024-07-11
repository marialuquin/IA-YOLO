# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:08:54 2024

@author: hdezl
"""

import cv2
import numpy as np

# =============================================================================
# # Cargar el modelo YOLO
# =============================================================================
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
layer_names = net.getLayerNames()

# =============================================================================
# Obtiene los nombres de todas las capas de la red.
# =============================================================================
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# =============================================================================
# Cargar nombres de clases
# =============================================================================
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# =============================================================================
# # Función para realizar la detección de objetos
# =============================================================================
def detect_objects(image_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Crear un blob y pasar la imagen por la red
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen a procesar
image_path = './Test_Images/test_image_3.jpg'

# Detectar objetos en la imagen
detect_objects(image_path)
