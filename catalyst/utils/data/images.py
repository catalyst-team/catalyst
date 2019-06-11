import os
import numpy as np
import cv2
import jpeg4py as jpeg


def read_image(image_name, datapath=None, grayscale=False):
    if datapath is not None:
        image_name = (
            image_name if image_name.startswith(datapath) else
            os.path.join(datapath, image_name)
        )

    img = None
    try:
        if image_name.endswith(("jpg", "JPG", "jpeg", "JPEG")):
            img = jpeg.JPEG(image_name).decode()
    except Exception:
        pass

    if img is None:
        img = cv2.imread(image_name)

        if len(img.shape) == 3:  # BGR -> RGB
            img = img[:, :, ::-1]

    if len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    if img.shape[-1] != 3 and not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img
