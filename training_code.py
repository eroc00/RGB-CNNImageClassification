#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from models import CNN, getHandMask, addHandMask
import sys
from Labels import label_dict

# Read in classifications if they exist
# Perform data normalization if needed (likely on input images)
# Add any other features to data processing stage like Edge Detection, Thresholding, etc...
# Split data into a training and test set
# Load all images into 2 lists: Training list, Test list 
# Initialize Sequential CNN Model
# 

def normalize(imgs, color_scale = "rgb"):
    if color_scale == "hsv":
        imgs[:, :, :, 1] /= 180
        imgs[:, :, :, 2] /= 255
        imgs[:, :, :, 3] /= 255
    
    else:
        imgs /= 255
    
    return imgs



train = False

if __name__ == "__main__":
   
    
    folder = "training_data"

    #file = f'{folder}/classifications_labels.xlsx'

    # Load all images in training_data folder
    imagepaths = []
    for root, dirs, files in os.walk(f"./{folder}/", topdown=False): 
        for name in files:
            path = os.path.join(root, name)
            if path.endswith("png"): # We want only the images
                imagepaths.append(path)

    # Prepare data
    X = []
    Y = []
    for path in imagepaths:
        img = cv.imread(path) # Reads image and returns np.array
        #img = cv.resize(img, (320, 120)) # Reduce image size so training can be faster
        X.append(addHandMask(img))

        # Processing label in image path
        category = path.split("/")[2]
        label = (category.split("\\")[1]) # Label name as a string
        label_val = list(label_dict.keys())[list(label_dict.values()).index(label)] # Get numerical representation of label
        
        Y.append([1 if i == label_val else 0 for i in range(len(label_dict.values()))])

    X = np.array(X, dtype='uint8')
    Y = np.array(Y)

    print(X.shape, Y.shape)
        
    # Read all images in folder 
    img_size = X[0].shape # 640x480 RGB image

    if train == True:
        # Split dataset into training and test data
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.15, random_state=1)
        print(trainX.shape, trainY.shape)

        # Initialize and train model
        model = CNN(img_size, len(label_dict.values()))
        model.init()

        epoch = 10
        print(f"Training model...")
        st = time.time()

        model.fit(trainX, trainY, batch_size=32,epochs=epoch,verbose=2)

        duration = time.time() - st
        print(f"It took {duration:.3f} seconds to train model in {epoch} epochs.")



        # Measure performance
        predictY = model.predict(testX)
        test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

        #cm = metrics.confusion_matrix(testY, predictY)

        print(f"Test Accuracy: {test_acc}")
        #print(cm)

        #Save model weights
        model.save("hand_classifier_1.keras")

    else:
        model = CNN(img_size, len(label_dict.values()))
        model.load("hand_classifier_1.keras")


    
    img = cv.imread(f"./{folder}/set0/Thumbs Up/0.png")
    pred_class = model.guess_gesture_class(img)

    result = cv.putText(img, label_dict[pred_class], (12, 48), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow("Capture", result)
    cv.waitKey()

    












"""
# ESSENTIAL: Input convolutional layer must be the same dimensions as the images 
model.add(Conv2D(filters=16,kernel_size=3,padding="same",activation="relu",input_shape=img_size))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.5))

# ESSENTIAL: Output is the anchor point and dimensions of the ROI shaped as a rectangle
# (cornerx, cornery, width, height)
model.add(Dense(4,activation='softmax'))
model.summary()

# Currently, the X and Y values may naturally take values within the range of the image's dimensions.
# This might cause an issue if the model predicts a point slightly off of its true value but is still acceptable
# Example: prediction = (4, 3), actual = (3, 3) would be considered acceptable parameters
# To solve that, the output points will be converted into a smaller range that covers several original points
# If a predicted point (PP) falls between a boundary of NxN with the actual point (AP) in the center, that is considered
#   as a correct prediction
# Same logic may apply to predicted height and width.
# NOTE: This approach is only for evaluation purposes; the real application will ideally use the raw PP. 

model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])



"""