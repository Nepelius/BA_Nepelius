import cv2 as cv

if __name__ == '__main__':
    capture = cv.VideoCapture("../data/Schafe.wmv")

    while True:
        success, img = capture.read()

        if success:
            cv.imshow("test", img)
            cv.waitKey(0)
        else:
            break