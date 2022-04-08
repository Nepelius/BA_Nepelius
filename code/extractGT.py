import cv2 as cv

if __name__ == '__main__':
    path = "C:\\Users\\Nepi\\Documents\\BA\\BA_Nepelius\\data\\Schafe.wmv"

    capture = cv.VideoCapture(path)

    rois = []

    while True:
        success, img = capture.read()

        if success:
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

    capture.release()