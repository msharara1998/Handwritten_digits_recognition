import cv2
import pickle
from scipy import ndimage
from math import ceil, floor
import numpy as np



#############################
# First Part: Preprocess Data
#############################

def getBestShift(img):

    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    m = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, m, (cols, rows))
    return shifted


def preprocess_image(img):
    """Takes an image nd-array as a parameter and uses some
    functions to make the necessary preprocessing so that it is
    ready to be fed into the network, and returns a preprocessed
    (784,1) nd-array"""
    
    # resize the images and invert it (black background)
    img = cv2.resize(255 - img, (28, 28))
    
    (thresh, img) = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)
 
    rows, cols = img.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        img = cv2.resize(img, (cols, rows))

    colsPadding = (int(ceil((28 - cols) / 2.0)), int(floor((28 - cols) / 2.0)))
    rowsPadding = (int(ceil((28 - rows) / 2.0)), int(floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    # reshaping image to fit to the network
    img = img.flatten() / 255.0
    img = img.reshape(784,1)
    return img



#####################################
# Second Part: Load the trained model
#####################################

trained_model = pickle.load(open('trained_model.sav', 'rb'))



#######################################
# Third Part: Image to Digit Classifier
#######################################

def image_to_digit(name):
    """Takes an image name as a parameter, opens the image as
    an nd-array, and then feeds it to the network to classify
    it. The image file should be in the same directory as that
    of main.py"""

    # opening the image as nd-array from directory
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    # preprocess the image
    img = preprocess_image(img)
    # feed the image to the model and get return the result
    result = trained_model.forward_propagation(img)
    return np.argmax(result)