import cv2
import numpy as np
from matplotlib import pyplot as plt

cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture Fotos")
img_counter = 0
path = "./lego_fotos/train/red/"


while True:
    ret, frame = cam.read()


    cv2.imshow("WebCam", frame)


    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print("Image Shape:", frame.shape )
        crop = frame[10:10, 20:20]
        print("Crop Shape:", crop.shape)


        #cv2.imshow('Image', crop)

        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, 255, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Mark corners', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        plt.imshow(frame), plt.show()

        img_name = path + "opencv_frame_{}.jpg".format(img_counter)
        #cv2.imwrite(img_name, frame)
        #print("{} written!".format(img_name))
        #img_counter += 1

cam.release()

cv2.destroyAllWindows()