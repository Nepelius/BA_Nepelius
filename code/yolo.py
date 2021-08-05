import cv2
import numpy as np

#inVid = '../data/Schafe.wmv'
inVid = 'vtest.avi'

capture = cv2.VideoCapture(inVid)
if not capture.isOpened:
    print('Unable to open')
    exit(0)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

np.random.seed(0)
colors = np.random.randint(0, 255, size=(len(classNames), 3)).tolist()

configPath = 'yolov3-tiny.cfg'
weights = 'yolov3-tiny.weights'

#net = cv2.dnn_DetectionModel(weights, configPath)
net = cv2.dnn.readNet(weights, configPath)
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#net.setInputSize(416,416)
#net.setInputScale(1/255.0)
#net.setInputMean((127.5,127.5,127.5))
#net.setInputSwapRB(True)
#net.setInputCrop(False)

while True:
    success, img = capture.read()
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    height, width, channels = img.shape

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

                # Rectangle
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # cv2.rectangle(img, (x,y), (x + w, y + h), colors[class_id], 2)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classNames[class_ids[i]])

            cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
            cv2.putText(img, label + ": " + str(round(confidences[i] * 100)) + "%", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_ids[i]], 2)
    #classIds, conf, bbox = net.detect(img, confThreshold=0.3)
    #print(classIds, bbox)
    #if len(classIds) != 0:
    #    for classId, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
    #        cv2.rectangle(img, box, color=colors[classId], thickness=2)
    #        cv2.putText(img, classNames[classId] + ": " + str(round(confidence * 100, 2)), (box[0], box[1] - 10),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[classId], 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)