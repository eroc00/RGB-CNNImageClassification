# Tool to determine appropriate image thresholding range

#from asyncio.windows_events import NULL
import numpy as np
import cv2 as cv
import sys

# Read file name from command line
filename = "training_data/cap.jpg"
print(filename)

# Image Processing
img = cv.imread(filename)
#img = cv.resize(img, (1280, 960))

# Convert image to desired color scale
origimage = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

# Histogram equalization if desired
#origimage[:, :, 0] = cv.equalizeHist(origimage[:, :, 0])
#origimage[:, :, 1] = cv.equalizeHist(origimage[:, :, 1])
#equalizedImg = cv.cvtColor(origimage, cv.COLOR_HSV2BGR)
#cv.imshow('Equalized Image', equalizedImg)

def nothing(x):
    print(x)
# Create a window named trackbars.
cv.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv.createTrackbar("L - Y", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - Cr", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - Cb", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U - Y", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - Cr", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - Cb", "Trackbars", 255, 255, nothing)


while True:
    # Get the new values of the trackbar in real time as the user changes 
    # them
    l_h = cv.getTrackbarPos("L - Y", "Trackbars")
    l_s = cv.getTrackbarPos("L - Cr", "Trackbars")
    l_v = cv.getTrackbarPos("L - Cb", "Trackbars")
    u_h = cv.getTrackbarPos("U - Y", "Trackbars")
    u_s = cv.getTrackbarPos("U - Cr", "Trackbars")
    u_v = cv.getTrackbarPos("U - Cb", "Trackbars")

    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])



 # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv.inRange(origimage, lower_range, upper_range)

    # You can also visualize the real part of the target color (Optional)
    res = cv.bitwise_and(img, img, mask=mask)
    
    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,img,res))

    cv.imshow('Trackbars', stacked)

    button = cv.waitKey(15)
    if button & 0xFF == ord('q'):
        print(lower_range, upper_range) # [ 92 131   0] [255 255 255]
        break

