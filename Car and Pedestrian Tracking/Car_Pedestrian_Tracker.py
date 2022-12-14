import cv2

#Our video
video = cv2.VideoCapture('carVideo.mp4')

#Pre-trained data on cars from opencv
car_classifier_file = 'car_detector.xml'
pedestrian_classifier_file = 'haarcascade_fullbody.xml'

# load some pre-trained data on cars from opencv
trained_car_data = cv2.CascadeClassifier(car_classifier_file)
trained_pedestrian_data = cv2.CascadeClassifier(pedestrian_classifier_file)

while True:

        #Read the current frame
        (read_succesful, frame) = video.read()

        if read_succesful:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break


        #detect cars in every frame of the video
        cars = trained_car_data.detectMultiScale(grayscaled_frame) 
        pedestrians = trained_pedestrian_data.detectMultiScale(grayscaled_frame)

        #cars will be in red squares
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #pedestrians will be in green squares
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Alberto Ramos Car Detector', frame)

        key = cv2.waitKey(1)

        #Stop if Q is pressed
        if key == 81 or key == 113:
            break

# Close the video
video.release()