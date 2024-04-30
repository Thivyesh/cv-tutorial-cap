import cv2
from transformers import pipeline
from PIL import ImageDraw
from PIL import Image
import numpy as np
import supervision as sv

#trenger å innstallere timm i python for å funke

#Initialize detection pipeline
checkpoint = "facebook/detr-resnet-50"
detector = pipeline(model=checkpoint, task="object-detection")

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

    # #Convert image to a likeable format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    
    #predict
    predictions = detector(
        frame,
        candidate_labels=["human face"]
    )
 
    # annotate frame
    draw = ImageDraw.Draw(frame)
    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")
    frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()