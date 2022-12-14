# PYTHON AI STARTER PROJECTS

This repo contains my first approach to Artificial Intelligence using Python. For these projects I followed an online tutorial and used the openCV libraries
to recognize and process the different shapes, as well as TensorFlow for the last project. These were my first programs:

## 1. Face Recognition
Using the haarcascade face recognition pre-trained data, I was able to recognize faces in a video stream. The program also detects the eyes and draws a rectangle around the faces. 

## 2. Car and Pedestrian Tracker
In this program, we used the same library as before, but pre-trained to recognize cars and pedestrians. The program detects the cars and pedestrians in a video stream and draws a red rectangle around the cars and a green rectangle around the pedestrians.

## 3. Smile Detector
This program again uses then openCV library, but this time it is pre-trained to recognize smiles. The program first detects the faces, draws a rectagle around them and then searchs inside that rectangle for a smile. If it finds a smile, it writes "smiling" below the face.

<br />
<br />
<br />
<br />

* How to run the programs:
1. Download the repository
2. Open the terminal and navigate to the folder where you downloaded the repository
3. Run the command: python3 "name of the file".py
4. The program will start running and you will see the video stream with the shape recognition. Press Q if you would like to exit the program.

* Requirements:
1. Python 3
2. OpenCV
3. Numpy
