import cv2
import numpy as np
import PySimpleGUI as ps
import threading

# GUI

ps.theme("DarkAmber")

file_frame = [
    [
        ps.Listbox(values=(1,2,3), size=(20,100))
    ]
]

video_viewer = [
    [
        #ps.Image(filename="", key="-IMAGE-")
        ps.Text("Press 'r' to pause/resume the video\nPress 's' to step\nPress 'esc' to exit")
    ],
    [
        ps.Button("Play", key="-PLAY-", enable_events=True)
    ]
]

layout = [
    [
        ps.Column(file_frame),
        ps.VSeparator(),
        ps.Column(video_viewer)
    ]
]

window = ps.Window("Animal detector", layout, size=(1525,780), location=(0,0))

# Program

inVid = '../data/Schafe.wmv'
#inVid = 'vtest.avi'


#if not capture.isOpened:
#    print('Unable to open')
#    exit(0)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

np.random.seed(0)
colors = np.random.randint(0, 255, size=(len(classNames), 3)).tolist()

configPath = 'yolov3-tiny.cfg'
weights = 'yolov3-tiny.weights'

net = cv2.dnn.readNet(weights, configPath)
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def play_video():
    capture = cv2.VideoCapture(inVid)
    stepping = False
    while True:
        success, img = capture.read()

        if success:
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                        # cv2.rectangle(img, (x,y), (x + w, y + h), colors[class_id], 2)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classNames[class_ids[i]])

                    cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                    cv2.putText(img, label + ": " + str(round(confidences[i] * 100)) + "%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_ids[i]], 2)

            # img = cv2.resize(img, (400,400), cv2.INTER_NEAREST)
            # imgbytes = cv2.imencode(".png", img)[1].tobytes()
            # window["-IMAGE-"].update(data=imgbytes)

            cv2.imshow("Output", img)


            if not stepping:
                key = cv2.waitKey(10)
            else:
                key = cv2.waitKey()
                stepping = False

            if key == 27:  # press 'esc' to exit
                cv2.destroyAllWindows()
                break
            if key == ord('r'):
                cv2.rectangle(img, (10,10), (20,40), (0,0,255),-1)
                cv2.rectangle(img, (30, 10), (40, 40), (0, 0, 255), -1)

                cv2.imshow("Output", img)
                cv2.waitKey()
            if key == ord('s'):
                stepping = True
                #cv2.waitKey()
        else:
            break

    capture.release()
# main loop

while True:
    event, values = window.read(timeout=20)
    if event == ps.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break

    #th = threading.Thread(target=play_video)
    if event == "-PLAY-":
        window.disable()
        play_video()
        window.enable()

window.close()
cv2.destroyAllWindows()