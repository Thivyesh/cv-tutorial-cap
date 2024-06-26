import cv2
from ultralytics import YOLO  # Import YOLO model from Ultralytics
import supervision as sv  # Import the supervision library for annotations

class ObjectDetectionWithWebcam:
    """
    This class performs real-time object detection using a webcam and YOLO model.

    Attributes:
        model (YOLO): YOLO object detection model.
        webcam (cv2.VideoCapture): Webcam object for capturing frames.
    """

    def __init__(self, model_weights: str = 'yolov8s.pt'):
        """
        Initializes the ObjectDetectionWithWebcam class.

        Args:
            model_weights (str): Path to the YOLO model weights file (default is 'yolov8s.pt').
        """
        self.model = YOLO(model_weights)
        
        # If windows use:
        self.webcam = cv2.VideoCapture(0)
        
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
            
            # Perform object detection on the frame using the YOLO model
            results = self.model(frame)[0]

            # Convert YOLO detections to Supervision Detections format
            detections = sv.Detections.from_ultralytics(results)

            # Create a bounding box annotator with specified thickness
            bounding_box_annotator = sv.BoundingBoxAnnotator(
                thickness=4
            )

            # Create a label annotator
            label_annotator = sv.LabelAnnotator()

            # Get labels for each detected object
            labels = [
                f"{self.model.model.names[class_name]} {confidence:.2f}"
                for class_name, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            # Annotate the frame with bounding boxes
            annotated_image = bounding_box_annotator.annotate(
                scene=frame, detections=detections)

            # Annotate the frame with labels
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)

            # Display the annotated frame
            cv2.imshow("Object Detection", annotated_image)

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