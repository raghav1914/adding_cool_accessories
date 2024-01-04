import cv2

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascase.xml')

# Load the bunny face image with an alpha channel
bunny_face = cv2.imread('bunny.png', cv2.IMREAD_UNCHANGED)

# Extract the bunny face and the alpha channel
bunny_face_image = bunny_face[:, :, :3]
bunny_face_mask = bunny_face[:, :, 3]

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Resize the bunny face to match the size of the detected face
        resized_bunny_face = cv2.resize(bunny_face_image, (w, h))
        resized_bunny_mask = cv2.resize(bunny_face_mask, (w, h))
        
        # Apply the bunny face mask to create a region of interest (ROI)
        roi = frame[y:y+h, x:x+w]
        roi_bunny = cv2.bitwise_and(resized_bunny_face, resized_bunny_face, mask=resized_bunny_mask)
        
        # Add the bunny face to the ROI
        bunny_face_final = cv2.add(roi, roi_bunny)
        
        # Update the frame with the bunny face
        frame[y:y+h, x:x+w] = bunny_face_final
        
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
