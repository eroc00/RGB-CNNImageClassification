# Read a video footage

# While stream is open:
#   display frame as image
#   Allow user to draw a box on the displayed image using the left mouse button
#   if desired ROI cannot be classified
#       press 'n' to indicate no box

# Save box information into an array
# Save array as an excel sheet using scikit-learn

import cv2 as cv
import numpy as np
import os
import pandas as pd
import sys

drawing = False
ix,iy = -1,-1

"""
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, rectangle
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
        cv.imshow(fn, img)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        displayedImg = np.copy(img)
        cv.rectangle(displayedImg, (ix, iy), (x, y), (255, 0, 0), 3)
        cv.imshow(fn, displayedImg)
        
        
        print((ix, iy, x-ix, y-iy))
        rectangle = [ix, iy, x-ix, y-iy]

"""

## MAIN ##

""" Classification categories
0) zero
1) one
2) two
3) three
4) four
5) five
8) confirm (thumbs up)
9) x
"""

folder = "training_data"

filenames = os.listdir(f"./{folder}")

# Read all images in folder 
imageList = []
for filename in filenames:
    f = filename.split(".")
    if len(f) > 1 and f[1] == "jpg":
        print(f"Loading image {f[0]}...")
        img = cv.imread(f"{folder}/{filename}")
        imageList.append((img, filename))
        
print("Loaded all images!")
classif = []
i = 0
while i < len(imageList):
    img, fn = imageList[i]

    # Bind event to window
    #cv.namedWindow(fn)
    #cv.setMouseCallback(fn, draw_rectangle)
    
    # Show image
    maskYCbCr = cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2YCrCb),np.array([92, 131, 0]), np.array([255, 255, 255]))  
    maskHSV = cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV), np.array([122, 0, 40]), np.array([179, 58, 255]))

    stacked = np.hstack((img,
                         cv.bitwise_and(img, img, mask=maskHSV),
                         cv.bitwise_and(img, img, mask=maskYCbCr)))

    cv.imshow(fn, stacked)

    # Wait for a signal
    sig = cv.waitKey()

    # Classify as "no detection" or reset already defined rectangle
    """
    if sig & 0xFF == ord('x'):
        print("Erased rectangle")
    """

    if sig & 0xFF == ord('n'):
        i += 1
        cv.destroyWindow(fn)

    if sig & 0xFF == ord('q'):
        cv.destroyAllWindows()
        exit()

    
    

classif = np.array(classif)
print(classif.shape)
dataFrame = pd.DataFrame()
dataFrame["imgID"] = classif[:, 0]
dataFrame["X"] = classif[:, 1]
dataFrame["Y"] = classif[:, 2]
dataFrame["Width"] = classif[:, 3]
dataFrame["Height"] = classif[:, 4]

dataFrame.to_excel(f"{folder}/classifications.xlsx")
print("Classified all images.")
