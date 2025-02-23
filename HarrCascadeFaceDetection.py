#DETECTS ONLY FRONTAL FACE
import cv2

# Load the Haar Cascade face detection model
face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the webcam
webCam = cv2.VideoCapture(0)

while True:
    has_frame, frame = webCam.read()  # Read frame from webcam

    if not has_frame:
        print("Error: Unable to capture video")
        break

    # Convert image to grayscale (improves detection performance)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    face_cor = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x1, y1, w, h) in face_cor:
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Display the output
    cv2.imshow("Face Detection", frame)

    # Press 'Enter' (key code 13) to exit
    if cv2.waitKey(10) == 13:
        break

# Release resources
webCam.release()
cv2.destroyAllWindows()
