import cv2


def cv2_read_image(image_path):
    image_array = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)   # (H, W, C(R, G, B)) (0~255) dtype = np.uint8
    return image_array