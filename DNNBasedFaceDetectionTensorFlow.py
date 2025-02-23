#DETECTS ONLY FRONTAL FACE
import cv2
import numpy as np

# Load the pre-trained DNN model
model_path = "opencv_face_detector_uint8.pb"
config_path = "opencv_face_detector.pbtxt"

neural_network = cv2.dnn.readNetFromTensorflow(model_path,config_path)


webCam = cv2.VideoCapture(0)

#the below size and means are default values that the openCV ssd model uses.
size = [300,300]
mean = [104, 177, 123]


while True:
    has_frame,frame = webCam.read()
    if not has_frame:
        break
    
    h,w = frame.shape[0],frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame,scalefactor=1,size=size,mean=mean,swapRB=False,crop=False)

    neural_network.setInput(blob)

    detections = neural_network.forward()

    # MORE ABOUT DETECTIONS IN COMMENTS BELOW
    # Loop through detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence score


        if confidence > 0.5:  # Filter weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # Scale box back to image size
            x1, y1, x2, y2 = box.astype("int")

            # Draw bounding box and confidence score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f"{confidence*100:.2f}%"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("DNN Face Detection", frame)

    # Press 'Enter' (key code 13) to exit
    if cv2.waitKey(10) == 13:
        break

############################################################
'''
DETECTIONS:

detections.shape has 4 dimensions:
scss
Copy
Edit
(1, 1, N, 7)
Where:
1 → Batch size (we process 1 image at a time).
1 → Number of detection networks (always 1).
N → Number of detected faces in the frame.
7 → Each detection contains 7 values (explained below).

Understanding detections[0, 0, i, :]
Each detected face contains 7 values, stored in the format:

Copy
Edit
detections[0, 0, i, :] = [0, 1, confidence, x_min, y_min, x_max, y_max]
Index	Value	Meaning
0	0	Reserved (not used)
1	1	Always 1 (object class ID)
2	confidence	Model’s confidence score (0 to 1)
3	x_min	Left boundary of the face (normalized)
4	y_min	Top boundary of the face (normalized)
5	x_max	Right boundary of the face (normalized)
6	y_max	Bottom boundary of the face (normalized)
'''
