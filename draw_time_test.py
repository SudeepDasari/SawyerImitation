import cv2
import matplotlib.pyplot as plt
from math import pi

img = cv2.imread('splashes/recording_oneshot.png')

cv2.putText(img, "Go!", (375, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 20,
                            cv2.LINE_AA)
plt.imshow(img[:, :, ::-1])
plt.show()