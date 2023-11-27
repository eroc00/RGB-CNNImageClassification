from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import cv2 as cv

checkpoint_path = "training/cp.ckpt"

def create_1D_model():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()])

  return model




def create_CNN_model(shape, num_classes):
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)

    model = keras.Sequential([
        keras.layers.Rescaling(1./1., input_shape=shape),
        keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

class CNN():

  
  def __init__(self, input_shape, output_size):
    self.input_shape = input_shape
    self.output_size = output_size
    self.model = None

  def init(self):
    self.model = Sequential([
      # ESSENTIAL: Input convolutional layer must be the same dimensions as the images 
      Conv2D(filters=16,kernel_size=3,padding="same",activation="relu",input_shape=self.input_shape),
      MaxPooling2D(pool_size=2),
      Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"),
      MaxPooling2D(pool_size=2),
      Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"),
      MaxPooling2D(pool_size=2),
      Dropout(0.2),
      Flatten(),
      Dense(200,activation='relu'),
      Dropout(0.5),

      # ESSENTIAL: Output is the anchor point and dimensions of the ROI shaped as a rectangle
      # (cornerx, cornery, width, height)
      Dense(self.output_size,activation='softmax')
    ])

    self.model.summary()
    self.model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])    


  def fit(self, trainX, trainY, batch_size=50,epochs=20,verbose=1):
     self.model.fit(trainX, trainY, batch_size,epochs,verbose)

  def predict(self, X, ver=0):
     return self.model.predict(X, verbose=ver) # takes slightly less than 100ms to perform prediction
  
  def evaluate(self, input, output, verbose=2):
     return self.model.evaluate(input, output, verbose)

  def guess_gesture_class(self, img, certainty_thr=0):
    input_img = np.zeros((1, img.shape[0], img.shape[1], (img.shape[2] + 1)), dtype='uint8')
    input_img[0, :, :, :] = addHandMask(img)
    prediction = self.predict(input_img)[0]
    certainty = np.max(prediction)

    if certainty < certainty_thr:
       return "None", prediction

    return np.argmax(prediction), certainty
  
  def save(self, filename):
     self.model.save(f"saved_models/{filename}")
  
  def load(self, filename):
     self.model = load_model(f"saved_models/{filename}")
     self.model.summary()
     

def getHandMask(img, lower_r=[120, 15, 70], higher_r=[255, 255, 255], stddev=2, thr=123, ero_size=2):
    lower_range = np.array(lower_r)
    upper_range = np.array(higher_r)

    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(imgHSV, lower_range, upper_range)

    mask = cv.GaussianBlur(mask, ksize=(0, 0), sigmaX=stddev, sigmaY=0) # sigmaX = 2, sigmaY = 0

    _, mask = cv.threshold(mask, thr, 255, cv.THRESH_BINARY)  # Thr (123, 255)  

    # Ideal dilation size: 5
    #dilation_size = 5
    #dilate_element = cv.getStructuringElement( cv.MORPH_ELLIPSE, (2*dilation_size + 1, 2*dilation_size + 1), (0, 0))

    # Ideal erosion size: 3
    erosion_size = ero_size
    erode_element = cv.getStructuringElement( cv.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size + 1), (0, 0))

    mask = cv.erode(mask, erode_element)
    #mask = cv.dilate(mask, dilate_element)

    return mask

def addHandMask(img):
    return np.concatenate((img, np.expand_dims(getHandMask(img), axis=2)), axis=2)