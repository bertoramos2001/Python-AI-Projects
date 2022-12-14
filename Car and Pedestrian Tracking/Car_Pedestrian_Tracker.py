import cv2

#Our video
video = cv2.VideoCapture('carVideo.mp4')

#Pre-trained data on cars from opencv
classifier_file = 'car_detector.xml'

# load some pre-trained data on cars from opencv
trained_car_data = cv2.CascadeClassifier(classifier_file)

while True:

        #Read the current frame
        (read_succesful, frame) = video.read()

        if read_succesful:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break


        #detect cars in every frame of the video
        cars = trained_car_data.detectMultiScale(grayscaled_frame) 

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('Alberto Ramos Car Detector', frame)

        cv2.waitKey(1)

# Close the video
video.release()