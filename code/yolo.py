import csv
import os
import cv2 as cv
import numpy as np
import PySimpleGUI as ps

# GUI

ps.theme("DarkGrey13")

file_frame = [
    [
        ps.Text("Load Video or folder"),
        ps.In(enable_events=True, key="-FOLDER-"),
        ps.FolderBrowse()
    ],
    [
        ps.Listbox(values=[], size=(70,20), enable_events=True, key="-FILE LIST-", select_mode="extended")
    ]
]

video_viewer = [
    [
        ps.Text("All animals count: \nBird count: \nCat count: \nDog count: \nHorse count: \n" +
                "Sheep count: \nCow count: \nElephant count: \nBear count: \n" +
                "Zebra count: \nGiraffe count: \n", key="-STATS-"),
    ],
    [
        ps.Text("Press 'r' to pause/resume the video\nPress 's' to step\nPress 'esc' to exit")
    ],
    [
        ps.Radio("Slow", key="-SLOW-", enable_events=True, group_id=0),
        ps.Radio("Fast", key="-FAST-", enable_events=True, group_id=0, default=True)
    ],
    [
        ps.Button("Play", key="-PLAY-", enable_events=True),
        ps.InputText(visible=False, enable_events=True, key='fig_path'),
        ps.FileSaveAs
        (
            button_text="Export data as...",
            key="-FILE SAVE-",
            file_types=(('CSV', '.csv'), ('TXT', '.txt')),
            enable_events=True
        )
    ]
]

layout = [
    [
        ps.Column(file_frame),
        ps.VSeparator(),
        ps.Column(video_viewer)
    ]
]

window = ps.Window("Animal detector", layout)

# Program

#inVid = '../data/Schafe.wmv'
#inVid = 'vtest.avi'
inVids = []

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

def ExportToCSV(filePath):
    f = open(filePath, 'w')
    writer = csv.writer(f, delimiter=";", lineterminator="\n")

    firstRow = ["", "", "All Animals", "Birds", "Cats", "Dogs", "Horses", "Sheeps", "Cows"]
    writer.writerow(firstRow)

    current_row = []
    frame_count = 0
    for i in range(len(staticValues.videos)):
        for k in range(staticValues.frames[i]):
            current_row.append(staticValues.videos[i])
            current_row.append("Frame: " + str(k + 1))
            for l in staticValues.animals[frame_count]:
                current_row.append(l)
            writer.writerow(current_row)
            frame_count += 1
            current_row = []

    f.close()

def IsValidBox(box):
    buffer = 75
    for tr in staticValues.tracking_boxes:
        if tr[0] - buffer < box[0] < tr[0] + buffer and tr[1] - buffer < box[1] < tr[1] + buffer and tr[2] - buffer < box[2] < tr[2] + buffer and tr[3] - buffer < box[3] < tr[3] + buffer:
            return False
    return True

def play_video(vid):
    staticValues.videos.append(vid)
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    print(classNames)

    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(len(classNames), 3)).tolist()

    configPath = ""
    weights = ""

    if staticValues.slowSelected:
        configPath = 'yolov3.cfg'
        weights = 'yolov3.weights'
    elif staticValues.fastSelected:
        configPath = 'yolov3-tiny.cfg'
        weights = 'yolov3-tiny.weights'

    net = cv.dnn.readNet(weights, configPath)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    capture = cv.VideoCapture(vid)
    stepping = False

    trackers = []

    current_conf = None
    current_id = None

    count_frames = 0

    staticValues.tracking_boxes = []

    while True:
        success, img = capture.read()

        if success:
            print(count_frames)
            count_frames += 1

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

                        #staticValues.AllAnimals += 1

                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        box = [x, y, w, h]
                        current_id = class_id
                        current_conf = float(confidence)
                        current_boxes.append(box)
                        confidences.append(current_conf)
                        class_ids.append(current_id)

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
                            #print(staticValues.AllAnimals)

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
                    cv.putText(img, label + ": " + str(int(round(confidences[i], 2) * 100)) + "%", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_ids[i]], 2)

            cv.imshow("Output", img)

            staticValues.animals.append([staticValues.AllAnimals, staticValues.Birds, staticValues.Cats, staticValues.Dogs,
                                         staticValues.Horses, staticValues.Sheeps, staticValues.Cows])

            if not stepping:
                key = cv.waitKey(10)
            else:
                key = cv.waitKey()
                stepping = False

            if key == 27:  # press 'esc' to exit
                cv.destroyAllWindows()
                break
            if key == ord('r'):
                cv.rectangle(img, (10,10), (20,40), (0,0,255),-1)
                cv.rectangle(img, (30, 10), (40, 40), (0, 0, 255), -1)

                cv.imshow("Output", img)
                cv.waitKey()
            if key == ord('s'):
                stepping = True
                #cv2.waitKey()
        else:
            break

    staticValues.frames.append(count_frames)
    capture.release()

# main loop GUI
while True:
    event, values = window.read(timeout=20)
    if event == ps.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break

    window["-STATS-"].update("All animals: " + str(staticValues.AllAnimals) + "\nBirds: " + str(staticValues.Birds) +
                             "\nCats: " + str(staticValues.Cats) + "\nDogs: " + str(staticValues.Dogs) +
                             "\nHorses: " + str(staticValues.Horses) + "\nSheeps: " + str(staticValues.Sheeps) +
                             "\nCows: " + str(staticValues.Cows) + "\n")

    if event == "-SLOW-":
        staticValues.slowSelected = True
        staticValues.fastSelected = False
    elif event == "-FAST-":
        staticValues.fastSelected = True
        staticValues.slowSelected = False

    if event == "-PLAY-":
        if inVids:
            for vid in inVids:
                window.disable()
                play_video(vid)
                window.enable()
        else:
            ps.popup_error("Please choose a file")

    if event == 'fig_path' and values['fig_path'] != '':
        print(values['-FILE SAVE-'])
        ExportToCSV(values['-FILE SAVE-'])

    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        names = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".wmv", ".avi", ".mp4"))
        ]

        window["-FILE LIST-"].update(names)

    if event == "-FILE LIST-":
        inVids = []
        try:
            for i in values["-FILE LIST-"]:
                filename = os.path.join(values["-FOLDER-"],i)
                inVids.append(filename)
        except:
            pass

window.close()
cv.destroyAllWindows()