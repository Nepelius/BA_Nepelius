import cv2 as cv

backSub = cv.createBackgroundSubtractorMOG2()

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
    gauss = cv.GaussianBlur(frame, (7,7), 0)

    # median filter
    medianFilter = cv.medianBlur(gauss, 7)
    fgMask = backSub.apply(medianFilter)

    cv.rectangle(medianFilter, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(medianFilter, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    #cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break