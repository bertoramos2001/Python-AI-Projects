import cv2

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#to capture video from webcam
webcam = cv2.VideoCapture(0) # (0) means it uses the default system camera

#iterate over the frames
while True:
    successful_frame_read, frame = webcam.read()

    #convert the frame to grayscale
    grayScaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayScaled_frame)

    #Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates: #we get x,y coordinate of top left and then width and height on each face we detect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #position top left, add width, add height, color green and thickness 2

    #Display the image with faces detected
    cv2.imshow('Alberto Ramos Face Detector', frame)
    key = cv2.waitKey(1)

    #stop is q is pressed
    if key == 81 or key == 113:
        break


print("Code Completed")
