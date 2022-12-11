import cv2

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
img = cv2.imread('elon.jpeg')

#convert the image to grayscale
grayScaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayScaled_img)

#Draw rectangles around the faces
(x, y, w, h) = face_coordinates[0] #we get x,y coordinate of top left and then width and height
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #position top left, add width, add height, color green and thickness 2

cv2.imshow('Alberto Ramos Face Detector', img)
cv2.waitKey()

print("Code Completed")
