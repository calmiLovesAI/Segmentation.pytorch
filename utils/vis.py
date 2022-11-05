import cv2
import numpy as np


def blend(foreground, background, weight):
    assert foreground.shape == background.shape
    foreground = np.uint8(foreground)
    blended = cv2.addWeighted(background, (1 - weight), foreground, weight, 0.0)
    return blended
