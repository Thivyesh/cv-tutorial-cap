import cv2
from transformers import pipeline
from PIL import ImageDraw, Image
import numpy as np

class ObjectDetectionWithWebcam:
    """
    This class performs real-time object detection using a webcam and DETR model.

    Attributes:
        detector: DETR object detection pipeline.
        webcam (cv2.VideoCapture): Webcam object for capturing frames.
    """

    def __init__(self, checkpoint: str = "facebook/detr-resnet-50"):
        """
        Initializes the ObjectDetectionWithWebcam class.

        Args:
            checkpoint (str): Name or path of the DETR checkpoint (default is "facebook/detr-resnet-50").
        """
        self.detector = pipeline(model=checkpoint, task="object-detection")
        # If windows use:
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # If Mac use:
        #self.webcam = cv2.VideoCapture(0)

        if not self.webcam.isOpened():
            raise RuntimeError("Cannot open webcam")

    def __del__(self):
        """
        Cleans up resources by releasing the webcam.
        """
        self.webcam.release()
        cv2.destroyAllWindows()

    def detect_objects(self):
        """
        Performs real-time object detection using the webcam and displays the annotated frames.
        """
        while True:
            # Read frame from webcam
            ret, frame = self.webcam.read()

            if not ret:
                print("Can't receive frame (stream end?), Exiting ...")
                break

            # Convert frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            # Predict objects in the frame
            predictions = self.detector(frame)

            # Annotate the frame with predicted bounding boxes and labels
            draw = ImageDraw.Draw(frame)
            for prediction in predictions:
                box = prediction["box"]
                label = prediction["label"]
                score = prediction["score"]

                xmin, ymin, xmax, ymax = box.values()
                draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")

            # Convert annotated frame back to OpenCV format
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display the annotated frame
            cv2.imshow("Object Detection", frame)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) == ord("q"):
                break

# Usage example:
if __name__ == "__main__":
    # Initialize ObjectDetectionWithWebcam class
    detector = ObjectDetectionWithWebcam()

    # Perform real-time object detection
    detector.detect_objects()
    detector.__del__()
