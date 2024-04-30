import torch
import cv2
import supervision as sv

#https://pytorch.org/hub/ultralytics_yolov5/
#https://supervision.roboflow.com/how_to/detect_and_annotate/

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

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
    results = model(frame)
    print(results)
    detections = sv.Detections.from_yolov5(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=4
    )
    label_annotator = sv.LabelAnnotator()

    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()