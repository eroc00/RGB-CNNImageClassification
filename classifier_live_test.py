import cv2 as cv
import models
import time
import numpy as np
from Labels import label_dict
import os


classifier = models.CNN((480,640,3), 4)
classifier.load("hand_classifier_1.keras")
print("Loaded model")

input_img = np.zeros((1, 480, 640, 3))

prediction_certainty = 0.60

#while halt is not True:
label_dict["None"] = "N/A"

cam = cv.VideoCapture(0)
print("Camera initialized")

halt = False
pause = False
while halt is not True:

    key = cv.waitKey(10)
    print(key)

    # Stop Video Catpure
    if key & 0xFF == ord('q'):
        halt = True

    # Pause Video
    elif key & 0xFF == ord('p'):
        pause = not pause

    if pause is True:
        continue

    res, img = cam.read()
    #print(img.shape)
    classification, certainty = classifier.guess_gesture_class(img, prediction_certainty)

    gesture = label_dict[classification]

    text = f"{gesture}: {certainty} sure"

    print(certainty)
        
    result = cv.putText(img, text, (12, 48), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    cv.imshow("Capture", img)


cv.destroyAllWindows()