import numpy as np
try:
    import keras_ocr
    _HAVE_KERAS_OCR = True
except Exception as _e:
    _HAVE_KERAS_OCR = False
    print(f"[text_removal] keras_ocr unavailable ({_e}); text removal will be a no-op")
import cv2

# Function to inpaint the text
def inpaint_text(image, pipeline):
    # Recognize text (and corresponding regions)
    prediction_groups = pipeline.recognize([image])

    # Define the mask for inpainting
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for box in prediction_groups[0]:
        # Extract the coordinates of the bounding box around the text
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        # Create a polygon that covers the text area more precisely
        polygon = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # Fill the polygon area on the mask
        cv2.fillPoly(mask, [polygon], 255)

    # Inpaint the image to remove text
    inpainted_img = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)

    return inpainted_img

# Class to handle text removal from images
class TextRemover:
    def __init__(self):
        self.pipeline = None
        if _HAVE_KERAS_OCR:
            try:
                self.pipeline = keras_ocr.pipeline.Pipeline()
            except Exception as e:
                print(f"[text_removal] keras_ocr pipeline init failed ({e}); text removal will be a no-op")
                self.pipeline = None

    def remove_text(self, img_path):
        if self.pipeline is None:
            img = cv2.imread(img_path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = keras_ocr.tools.read(img_path)
        img_text_removed = inpaint_text(img, self.pipeline)
        return img_text_removed

