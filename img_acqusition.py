import cv2 as cv
import numpy as np
from datetime import datetime
import os, sys
from Labels import label_dict, label_num
from models import getHandMask

drawing = False
ix,iy = -1,-1

imgsPerClassification = 30

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix = min(x, 640)
        iy = min(y, 480)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
       #cv.rectangle(img, (ix, iy),(x, y),(0, 255, 255),-1)
        #print(type(mask), mask[iy, ix])

def segmentImage(img):
    # Get the new values of the trackbar in real time as the user changes 
    # them
    l_h = cv.getTrackbarPos("L - H", "Capture")
    l_s = cv.getTrackbarPos("L - S", "Capture")
    l_v = cv.getTrackbarPos("L - V", "Capture")
    u_h = cv.getTrackbarPos("U - H", "Capture")
    u_s = cv.getTrackbarPos("U - S", "Capture")
    u_v = cv.getTrackbarPos("U - V", "Capture")
    k = cv.getTrackbarPos("Gaussian Blur", "Capture")
    thr = cv.getTrackbarPos("Threshold", "Capture")

    dilation_size = cv.getTrackbarPos("Dilation", "Capture")
    erosion_size = cv.getTrackbarPos("Erosion", "Capture")


    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    mask = getHandMask(img, lower_r=lower_range, higher_r=upper_range, stddev=k, thr=thr, ero_size=erosion_size)

    #imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Histogram equalization to combat lighting issues
    #imgHSV[:, :, 1] = cv.equalizeHist(imgHSV[:, :, 1])
    #imgHSV[:, :, 2] = cv.equalizeHist(imgHSV[:, :, 2])


    #mask = cv.inRange(imgHSV, lower_range, upper_range)

    # Ideal erosion size: 3
    #erode_element = cv.getStructuringElement( cv.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size + 1), (0, 0))
    #mask = cv.erode(mask, erode_element)

    #mask = cv.GaussianBlur(mask, ksize=(0, 0), sigmaX=max(1, k), sigmaY=0) # sigmaX = 2, sigmaY = 0

    #_, mask = cv.threshold(mask, thr, 255, cv.THRESH_BINARY)  # Thr (123, 255)  


    # Ideal dilation size: 5
    #dilate_element = cv.getStructuringElement( cv.MORPH_ELLIPSE, (2*dilation_size + 1, 2*dilation_size + 1), (0, 0))
    #mask = cv.dilate(mask, dilate_element)

    #print(type(mask), mask[iy, ix])


    # You can also visualize the real part of the target color (Optional)
    res = cv.bitwise_and(img, img, mask=mask)
    
    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,img,res))

    stacked = cv.putText(stacked, f"{label_dict[label]} Gesture #{imgCount}/{imgsPerClassification}", (12, 48), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow("Capture", stacked)

def mode2(img):
    print("Mode 2")


########################### MAIN ###############################
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Provide a folder name in the command line argument to save acquired pictures.")
        exit()

    # Create a window
    cv.namedWindow("Capture")

    # Bind event to window
    cv.setMouseCallback("Capture", draw_rectangle)

    def nothing(x):
        print(x)

    # Now create 6 trackbars that will control the lower and upper range of 
    # H,S and V channels. The Arguments are like this: Name of trackbar, 
    # window name, range,callback function. For Hue the range is 0-179 and
    # for S,V its 0-255.
    cv.createTrackbar("L - H", "Capture", 120, 179, nothing)
    cv.createTrackbar("L - S", "Capture", 15, 255, nothing)
    cv.createTrackbar("L - V", "Capture", 70, 255, nothing)
    cv.createTrackbar("U - H", "Capture", 179, 179, nothing)
    cv.createTrackbar("U - S", "Capture", 255, 255, nothing)
    cv.createTrackbar("U - V", "Capture", 255, 255, nothing)
    cv.createTrackbar("Gaussian Blur", "Capture", 1, 9, nothing)
    cv.createTrackbar("Threshold", "Capture", 0, 255, nothing)


    cv.createTrackbar("Dilation", "Capture", 5, 9, nothing)
    cv.createTrackbar("Erosion", "Capture", 3, 9, nothing)


    folder = "training_data/" + sys.argv[1] # Name of folder in training_data folder
    if not os.path.exists(f'{folder}'):
        os.mkdir(f'{folder}')
        for i in label_dict.keys():
            os.mkdir(f'{folder}/{label_dict[i]}')


    cam = cv.VideoCapture(0)
    halt = False
    pause = False
    imgCount = 0
    opMode = 0
    label = 0
    while halt is not True:

        ############# MODE SWITCHES ########################
        key = cv.waitKey(10)
        #print(key)

        # Stop Video Catpure
        if key & 0xFF == ord('q'):
            halt = True

        # Capture Image (c)
        elif key & 0xFF == ord('c'):
            imgName = f"{folder}/{label_dict[label]}/{str(imgCount)}.png"
            cv.imwrite(imgName, img)
            imgCount += 1
            print(f"Image {imgCount} saved")
            if imgCount == imgsPerClassification:
                label += 1
                imgCount = 0
            
            if label > label_num:
                halt = True
                break

        # Pause Video
        elif key & 0xFF == ord('p'):
            pause = not pause
        
        elif key & 0xFF == ord('s'):
            opMode = (opMode + 1) % 2


        ################## MAIN LOOP ######################
        
        # Pause video
        if pause is True:
            continue

        res, img = cam.read()
        mask = np.zeros((img.shape[0], img.shape[1], 1))

        # Image Segmentation Mode
        if opMode == 0:
            segmentImage(img)

        elif opMode == 1:
            mode2(img)

    print("Completed image acquisition/classification")

    # YCrCb masks
    # Low lighting mask: [141 134   0] [255 255 255]
    # High lighting mask: [131 111 127] [255 255 255]

    #print(lower_range, upper_range) # [ 120, 15, 80] [255, 255, 255]

    cv.destroyAllWindows()

    # (335, 112, 262, 340)