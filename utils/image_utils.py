import cv2


class FaceDetector():
    """The ``FaceDetector`` class construct a detector to looking for faces in
    the images. Algorithm used is OpenCV.
    """

    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            'utils/haarcascade_frontalface_default.xml')

    def detect(self, image):
        """Returns the position of the bigger face in the image.

        Args:
            image (:obj:`numpy.ndarray`): input image.

        Returns:
            tuple: a tuple containing the top left coordinates and dimensions of
                the face found (x, y, w, h). Or None if no face was found in the
                image.
        """
        # Coordinate of the face found
        x1 = x2 = y1 = y2 = 0

        # Convert color image to grey scale
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Set minimal face size
        img_size = (image.shape[0] + image.shape[1]) // 2
        size_min = img_size // 8
        # Detect faces
        rects = self.detector.detectMultiScale(gray_img,
                                               scaleFactor=1.1,
                                               minNeighbors=4,
                                               minSize=(size_min, size_min))
        # Keep the bigger face if several faces are found
        for (x, y, w, h) in rects:
            if w + h > x2 - x1 + y2 - y1:
                x1 = x
                x2 = x + w
                y1 = y
                y2 = y + h

        if x2 - x1 > 0:
            return (x1, y1, x2 - x1, y2 - y1)

        return None
