import cv2

#img = cv2.imread('dog.jpg')

inVid = '../data/Schafe.wmv'
#inVid = 'vtest.avi'

capture = cv2.VideoCapture(inVid)
if not capture.isOpened:
    print('Unable to open')
    exit(0)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

configPath = 'yolov3-tiny.cfg'
weights = 'yolov3-tiny.weights'

net = cv2.dnn_DetectionModel(weights, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, img = capture.read()
    classIds, conf, bbox = net.detect(img, confThreshold=0.3)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)