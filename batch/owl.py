import cv2
import skimage.data
from transformers import pipeline
from PIL import ImageDraw
from PIL import Image
import numpy as np

import skimage
import supervision as sv

#Initialize detection pipeline
checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
image = skimage.data.astronaut()
image = Image.fromarray(np.uint8(image)).convert("RGB")
#predict
predictions = detector(
    image,
    candidate_labels=["face", "rocket", "nasa badge", "star-spangled banner"]
)
print(predictions)
# detections = sv.Detections.from_transformers(predictions)
 
# bounding_box_annotator = sv.BoundingBoxAnnotator(
#     thickness=4
# )


#annotate frame
draw = ImageDraw.Draw(image)
for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]

    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")
frame = np.array(image)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

cv2.imshow("frame", frame)
cv2.waitKey(0)
if cv2.waitKey(1) == ord("q"):
    cv2.destroyAllWindows()