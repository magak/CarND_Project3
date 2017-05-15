import cv2
def preproc(img):
    img = cv2.resize(img,(160,80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    