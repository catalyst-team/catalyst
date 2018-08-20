import os
import cv2
import jpeg4py as jpeg
import numpy as np


class ImageReader(object):
    def __init__(self, row_key, dict_key, datapath=None, grayscale=False):
        self.row_key = row_key
        self.dict_key = dict_key
        self.datapath = datapath
        self.grayscale = grayscale

    def __call__(self, row):
        image_name = str(row[self.row_key])

        if self.datapath is not None:
            image_name = (
                image_name
                if image_name.startswith(self.datapath)
                else os.path.join(self.datapath, image_name))

        img = None
        try:
            if image_name.endswith(("jpg", "JPG", "jpeg", "JPEG")):
                img = jpeg.JPEG(image_name).decode()
        except:
            pass

        if img is None:
            img = cv2.imread(image_name)

            if len(img.shape) == 3:  # BGR -> RGB
                img = img[:, :, ::-1]

        if len(img.shape) < 3:  # grayscale
            img = np.expand_dims(img, -1)

        if img.shape[-1] != 3 and not self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        result = {self.dict_key: img}
        return result


class ScalarReader(object):
    def __init__(
            self, row_key, dict_key, dtype=np.float32, default_value=None):
        self.row_key = row_key
        self.dict_key = dict_key
        self.dtype = dtype
        self.default_value = default_value

    def __call__(self, row):
        scalar = self.dtype(row.get(self.row_key, self.default_value))
        result = {self.dict_key: scalar}
        return result


class TextReader(object):
    def __init__(self, row_key, dict_key, encode_fn=lambda x: x):
        self.row_key = row_key
        self.dict_key = dict_key
        self.encode_fn = encode_fn

    def __call__(self, row):
        text = row[self.row_key]
        text = self.encode_fn(text)
        result = {self.dict_key: text}
        return result


class ReaderCompose(object):
    def __init__(self, readers, mixins=None):
        self.readers = readers
        self.mixins = mixins or []

    def __call__(self, row):
        result = {}
        for fn in self.readers:
            result = {**result, **fn(row)}
        for fn in self.mixins:
            result = {**result, **fn(result)}
        return result
