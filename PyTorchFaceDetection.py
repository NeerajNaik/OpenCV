#DETECTS BOTH ALL ANGLES OF FACES
from facenet_pytorch import MTCNN
import cv2

# Load MTCNN face detector
mtcnn = MTCNN(keep_all=True)

# Read image
webCam = cv2.VideoCapture(0)

while True:

    has_frame,frame = webCam.read()

    if not has_frame:
        break
    # Convert to RGB (MTCNN requires RGB images)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(image_rgb)
    
    if boxes is not None and len(boxes) > 0:
        # Draw boxes on the image
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Face Detection", frame)

    if(cv2.waitKey(1) == ord('e')):
        break

webCam.release()
cv2.destroyAllWindows()
