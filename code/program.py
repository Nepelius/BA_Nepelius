import cv2 as cv

backSub = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

#inVideo = "vtest.avi"
inVideo = "../data/Schafe.wmv"

capture = cv.VideoCapture(inVideo)
if not capture.isOpened:
    print('Unable to open')
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # gauß filter
    # gauss = cv.GaussianBlur(frame, (5,5), 0)

    # median filter
    # medianFilter = cv.medianBlur(gauss, 5)
    fgMask = backSub.apply(frame)
    _, fgMask = cv.threshold(fgMask, 254, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # get rid of small contour areas
        area = cv.contourArea(cnt)
        detections = []
        if area > 200:
            # draw contours on original video
            # cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            # green rectangles
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.rectangle(fgMask, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(fgMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))



    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break