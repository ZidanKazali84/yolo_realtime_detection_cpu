import cv2
import numpy as np
import time
import os
#include <opencv2/opencv.hpp>

# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers=[]
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i-1])
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = 0
frame_id = 0
backGround = cv2.imread('jpg/pokedex.png')
while True:
    _, frame = cap.read()
    frame_id = 20
    

    
    height, width, channels = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            nama = label[1:10] + '.png'
            print(nama)
            if nama in os.listdir("resources\deskripsi"):
                #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, "Terdeteksi:", (10,340), font, 3, (239, 255, 255), 3)
                gambar = 'resources\AR' + label+ '.png'
                teks = 'resources\deskripsi' + label+ '.png'
                img = cv2.imread(gambar)
                img = cv2.resize(img, None, fx=0.4, fy=0.4)
                img = cv2.resize(img, (325,265))
                height, width, channels = img.shape
                data = cv2.imread(teks)
                data = cv2.resize(data, None, fx=0.4, fy=0.4)
                data = cv2.resize(data, (280, 440))
                height, width, channels = data.shape
                frame[10:10 + 440, 350:350 + 280]  = data
                frame[15:15 + 265, 10:10 + 325]  = img
                print(gambar)
                cv2.putText(frame, label[1:10], (50,420), cv2.FONT_HERSHEY_TRIPLEX, 2, (239, 255, 255), 3)
                


    background_resized = cv2.resize(backGround, (384, 600))
    
    
    #crop_frame = frame[0:400, 120:380] #Sesuaikan koordinat cropping sesuai dengan kebutuhan

    # Mengubah ukuran subset frame webcam menjadi 256x400
    frame = cv2.resize(frame, (350, 245 ))
    x = 17  # Jarak dari kiri
    y = 19  # Jarak dari atas

    # Menentukan batas koordinat frame yang akan dimasukkan ke latar belakang
    x_start = x
    y_start = y
    x_end = x + frame.shape[1]
    y_end = y + frame.shape[0]
    # Memasukkan frame ke latar belakang
    
    background_resized[y_start:y_end, x_start:x_end] = frame
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    starting_time = time.time()
    cv2.putText(background_resized, "FPS: " + str(int(fps)), (105, 362), font, 3, (255, 0, 0), 3)
    cv2.putText(background_resized, str(round(confidence*100,1))+"%", (146, 524), font, 2, (255, 0, 0), 3)
    cv2.imshow("BG", background_resized)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
