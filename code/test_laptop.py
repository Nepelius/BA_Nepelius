import cv2 as cv
import numpy as np

class staticValues:
    animals = []
    frames = []
    videos = []
    tracking_boxes = []
    fastSelected = True
    slowSelected = False
    AllAnimals = 0
    Birds = 0
    Cats = 0
    Dogs = 0
    Horses = 0
    Sheeps = 0
    Cows = 0

def IsValidBox(box):
    buffer = 75
    for tr in staticValues.tracking_boxes:
        if tr[0] - buffer < box[0] < tr[0] + buffer and tr[1] - buffer < box[1] < tr[1] + buffer and tr[2] - buffer < box[2] < tr[2] + buffer and tr[3] - buffer < box[3] < tr[3] + buffer:
            return False
    return True

if __name__ == '__main__':
    capture = cv.VideoCapture("../data/data_Schafe.wmv")

    trackers = []
    staticValues.tracking_boxes = []

    stepping = False

    # Get classnames
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # random color for each class
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(len(classNames), 3)).tolist()

    configPath = 'yolov3.cfg'
    weights = 'yolov3.weights'

    net = cv.dnn.readNet(weights, configPath)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        success, img = capture.read()

        if success:

            blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(outputlayers)

            height, width, channels = img.shape

            current_boxes = []
            class_ids = []
            confidences = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # center point
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        # width, height
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # upper left point
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        box = [x, y, w, h]
                        current_conf = float(confidence)
                        current_boxes.append(box)
                        confidences.append(current_conf)
                        class_ids.append(class_id)

                        if IsValidBox(box):
                            # Object tracking

                            if classNames[class_id] == "bird":
                                staticValues.Birds += 1
                                staticValues.AllAnimals += 1
                                print(classNames[class_id] + ": " + str(staticValues.Birds))
                            elif classNames[class_id] == "cat":
                                staticValues.Cats += 1
                                staticValues.AllAnimals += 1
                                print(classNames[class_id] + ": " + str(staticValues.Cats))
                            elif classNames[class_id] == "dog":
                                staticValues.Dogs += 1
                                staticValues.AllAnimals += 1
                                print(classNames[class_id] + ": " + str(staticValues.Dogs))
                            elif classNames[class_id] == "horse":
                                staticValues.Horses += 1
                                staticValues.AllAnimals += 1
                                print(classNames[class_id] + ": " + str(staticValues.Horses))
                            elif classNames[class_id] == "sheep":
                                staticValues.Sheeps += 1
                                staticValues.AllAnimals += 1
                                print(classNames[class_id] + ": " + str(staticValues.Sheeps))
                            elif classNames[class_id] == "cow":
                                staticValues.Cows += 1
                                staticValues.AllAnimals += 1
                                print(classNames[class_id] + ": " + str(staticValues.Cows))

                            tracker = cv.TrackerCSRT_create()
                            tracker.init(img, box)
                            trackers.append(tracker)

            # update trackers every frame
            for t in trackers:
                su, bbox = t.update(img)
                if su:
                    staticValues.tracking_boxes.append(bbox)
                else:
                    trackers.remove(t)


            # Non-Maximum supression to reduce number of boxes
            indexes = cv.dnn.NMSBoxes(current_boxes, confidences, 0.5, 0.4)
            # loop through all boxes on current image and draw them
            for i in range(len(current_boxes)):
                if i in indexes:
                    x, y, w, h = current_boxes[i]
                    label = str(classNames[class_ids[i]])

                    cv.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                    cv.putText(img, label + ": " + str(int(round(confidences[i], 2) * 100)) + "%", (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_ids[i]], 2)

            cv.imshow("Output", img)

            # stepping video
            if not stepping:
                key = cv.waitKey(10)
            else:
                key = cv.waitKey()
                stepping = False

            if key == 27:  # press 'esc' to exit
                cv.destroyAllWindows()
                break

            # stop / resume video
            if key == ord('r'):
                # red rectangle
                cv.rectangle(img, (10,10), (20,40), (0,0,255),-1)
                cv.rectangle(img, (30, 10), (40, 40), (0, 0, 255), -1)

                cv.imshow("Output", img)
                cv.waitKey()

            if key == ord('s'):
                stepping = True
        else:
            break