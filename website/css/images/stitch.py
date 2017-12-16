import cv2
import glob
import numpy as np
IMAGE_WIDTH = 759
IMAGE_HEIGHT = 506

image_paths = glob.glob("./*.jpg")
images = [cv2.resize(cv2.imread(i), (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA) for i in image_paths]
stitch = np.concatenate(images, axis = 1)
print 'stitch shape', stitch.shape
cv2.imwrite('stitch.jpg', stitch)