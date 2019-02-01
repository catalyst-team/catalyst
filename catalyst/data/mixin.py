import cv2


class ImageHistogramMixin:
    """
    Calculates color histogram for images in dataset
    """

    def __init__(self, input_key: str, output_key: str):
        """
        Args:
            input_key (str): input key to use from annotation dict
            output_key (str): output key to use to store the result
        """
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, row):
        """Reads a row from your annotations dict and
        calculates color histograms

        Args:
            row: elem in your dataset.

        Returns:
            Dictionary with color hists
        """
        img = row[self.input_key]

        result = {}
        for channel in range(img.shape[2]):
            hist_ = cv2.calcHist([img], [channel], None, [256], [0, 256])
            hist_key = self.output_key + f"_{channel:{1}}"
            result[hist_key] = hist_.ravel()

        return result
