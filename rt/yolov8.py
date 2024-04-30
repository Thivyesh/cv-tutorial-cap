import cv2
from ultralytics import YOLO
import supervision as sv


#Initialize yolo model
model = YOLO('yolov8s.pt')
#Initialize webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame =  cap.read()

    if not ret:
        print("Can't receive frame (stream end?), Exiting ...")
        break
    
    #predict
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=4
    )
    label_annotator = sv.LabelAnnotator()
    detections = detections[detections.class_id !=0]
    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    cv2.imshow("frame", annotated_image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()