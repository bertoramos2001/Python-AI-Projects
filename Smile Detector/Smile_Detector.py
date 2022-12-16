import cv2

#Pre-trained frontal face and smile data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    # Read webcam footage
    (successful_frame_read, frame) = webcam.read()

    # If there is an error, abort
    if not successful_frame_read:
        break

    # Convert to black and white image
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = trained_face_data.detectMultiScale(frame_grayscale)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

        the_face = frame[y:y+h, x:x+w] # from the 2D array, get both dimensions from x and y to the limits
        the_face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect the smiles
        smiles = trained_smile_data.detectMultiScale(the_face_grayscale, scaleFactor=1.7, minNeighbors=20)
        #for (x2, y2, w2, h2) in smiles:
            #cv2.rectangle(the_face, (x2, y2), (x2+w2, y2+h2), (255, 0, 255), 5)
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 255, 0))

    # Show webcam footage
    cv2.imshow('Alberto Ramos Smile Detector', frame)

    #read the key each milisecond
    key = cv2.waitKey(1)

    #Stop if Q is pressed
    if key == 81 or key == 113:
        break

webcam.release()