import cv2

import urllib.request
import numpy as np


# Download the YOLO weights and configuration files
urllib.request.urlretrieve('https://pjreddie.com/media/files/yolov3.weights', 'yolo_weights.weights')
urllib.request.urlretrieve('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', 'yolo.cfg')




# Load the YOLO object detector
net = cv2.dnn.readNet('yolo_weights.weights', 'yolo.cfg')

# Load the image
img = cv2.imread('satellite_image.jpg')

# Define the class labels
classes = ['person', 'car', 'truck', 'motorbike']

# Set the input image size for the network
input_size = (416, 416)

# Preprocess the image for the network
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, input_size, swapRB=True, crop=False)

# Set the input for the network
net.setInput(blob)

# Perform forward pass and get the output from the network
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Define the confidence threshold and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Loop over the outputs from the network and filter detections
detections = []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold and classes[class_id] in classes:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])
            x = center_x - width // 2
            y = center_y - height // 2
            detections.append([x, y, width, height, confidence, class_id])

confidences = [detection[5] for detection in detections]


# Apply non-maximum suppression to remove redundant detections
indices = cv2.dnn.NMSBoxes(detections, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes around the detected objects
for i in indices:
    i = i[0]
    x, y, w, h = detections[i][:4]
    label = classes[detections[i][5]]
    confidence = detections[i][4]
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    text = '{}: {:.2f}'.format(label, confidence)
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with the detections
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
