import cv2
import numpy as np
import time
import os

# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load background image
backGround = cv2.imread('jpg/pokedex.png')

# Initialize timers and intervals
detection_timer = time.time()
pause_timer = time.time()
detection_interval = 180  # Deteksi setiap 3 menit
pause_interval = 300  # Jeda 5 menit setelah deteksi

font = cv2.FONT_HERSHEY_PLAIN
frame_id = 0

# Webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Check if it's time to perform object detection
    if time.time() - detection_timer >= detection_interval:
        # Detect objects using YOLO and OpenCV
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Reset the pause timer
        pause_timer = time.time()

        # Showing information on the screen
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
                nama = label[1:10] + '.png'
                print(nama)
                if nama in os.listdir("resources/deskripsi"):
                    cv2.putText(frame, "Terdeteksi:", (10, 340), font, 3, (255, 0, 0), 3)
                    gambar = 'resources/AR' + label + '.png'
                    teks = 'resources/deskripsi' + label + '.png'
                    img = cv2.imread(gambar)
                    img = cv2.resize(img, (325, 265))
                    data = cv2.imread(teks)
                    data = cv2.resize(data, (280, 440))
                    frame[10:10 + 440, 350:350 + 280] = data
                    frame[15:15 + 265, 10:10 + 325] = img
                    print(gambar)
                    cv2.putText(frame, label[1:10], (50, 420), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 3)

    # Check if it's within the pause interval
    if time.time() - pause_timer >= pause_interval:
        # Reset the detection timer
        detection_timer = time.time()

    # Resize and insert frame into the background image
    frame = cv2.resize(frame, (350, 245))
    x = 17  # Distance from the left
    y = 19  # Distance from the top
    x_start = x
    y_start = y
    x_end = x + frame.shape[1]
    y_end = y + frame.shape[0]
    backGround[y_start:y_end, x_start:x_end] = frame

    elapsed_time = time.time() - detection_timer
    fps = frame_id / elapsed_time
    cv2.putText(backGround, "FPS: " + str(int(fps)), (105, 362), font, 3, (255, 0, 0), 3)
    cv2.imshow("BG", backGround)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
