import cv2 
import tkinter as tk
from tkinter import messagebox  

# Load the required trained XML classifiers for face and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Capture frames from the default camera
cap = cv2.VideoCapture(0) 

# Function to display a pop-up message
def show_popup():
    # Create a tkinter root window and hide it 
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showwarning("Alert", "Please focus your eyes on the screen!") 
    root.destroy()  # Destroy the hidden root window

# Loop runs if capturing has been initialized
while True:
    # Read frames from the camera
    ret, img = cap.read() 

    # Convert to grayscale for the face and eyes detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Track whether eyes are detected or not
    eyes_detected = False

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Region of interest for face (both color and grayscale)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w] 

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            eyes_detected = True  # If eyes are detected, set this to True
        else:
            eyes_detected = False  # If no eyes are detected, set to False

        # Draw a rectangle around each eye
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # If no eyes are detected, show a pop-up alert
    if not eyes_detected and len(faces) > 0:
        show_popup()

    # Display the resulting image with the face and eye detection rectangles
    cv2.imshow('img', img)

    # Exit the loop when 'Esc' key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
