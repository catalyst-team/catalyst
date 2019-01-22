import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def bytes2cv_image(bytes):
    img = np.fromstring(bytes, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)[:, :, ::-1]
    return img


def cv_image2bytes(img):
    _, img_data = cv2.imencode(".png", img)
    bytes = img_data.astype(np.uint8).tobytes()
    return bytes
