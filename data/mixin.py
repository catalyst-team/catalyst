import cv2


class ImageHistogramMixin(object):
    def __init__(self, dict_img_key, dict_histogram_key):
        self.dict_img_key = dict_img_key
        self.dict_histogram_key = dict_histogram_key

    def __call__(self, dict_):
        img = dict_[self.dict_img_key]

        result = {}
        for channel in range(img.shape[2]):
            hist_ = cv2.calcHist(
                [img], [channel], None, [256], [0, 256])
            hist_key = self.dict_histogram_key + f"_{channel:{1}}"
            result[hist_key] = hist_.ravel()

        return result
