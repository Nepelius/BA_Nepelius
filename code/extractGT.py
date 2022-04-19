import cv2 as cv

if __name__ == '__main__':
    path = "C:\\Users\\nepel\\MBI Bachelor\\Semester 5\\Bachelorarbeit\\Videos\\Untersulzbachtal\\Gams.wmv"

    capture = cv.VideoCapture(path)

    rois = []

    img_count = 0

    while True:
        success, img = capture.read()

        if success:
            img_count += 1
            #out = cv.selectROIs("Output", img)

            #if len(out) != 0:
                #for o in out:
                    #x = o[0]
                    #y = o[1]
                    #w = o[2]
                    #h = o[3]
                    #rois.append([x, y, w, h])
                    #print(rois)

            cv.imshow("Output", img)
            cv.waitKey()
        else:
            break


    print(img_count)
    capture.release()