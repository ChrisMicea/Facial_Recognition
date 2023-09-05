"""
Usage:
  face_recognize_run.py
  
Options:
  -h, --help                     Show this help
"""
  
# importing libraries
import face_recognition
import cv2
import sys
import docopt
from sklearn import svm
import os
from array import array
import numpy as np
import joblib

# define a video capture object
encodings = []
names = []

def face_recognize_antrenat(test, encodings, names):
    # Create and train the SVC classifier
    clf = svm.SVC(gamma ='scale')
    clf.fit(encodings, names)

    # Load the test image with unknown faces into a numpy array
    #test_image = face_recognition.load_image_file(test)
    test_image = test
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    cv2.putText(test ,"Number of faces detected: " + str(no), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 0), 2)
    #print()
  
    # Predict all the faces in the test image using the trained classifier
    #print("Found:")
    cv2.putText(test ,"Found:", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 0), 2)
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        #print(*name)
        cv2.putText(test , *name, (100 + i * 100, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 69, 0), 2)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(test, (left, top), (right, bottom), (255, 69, 0), 2)
    cv2.putText(test ,"Press 'esc' to exit", (450, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.imshow('Video Footage', test)

def main():
    args = docopt.docopt(__doc__)
    #read encodings from file
    encodings = joblib.load("encodings.bin")
    names = joblib.load("names.bin")
    vid = cv2.VideoCapture(0)
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        #cv2.imshow('Video Footage', frame)

        # the 'esc' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord(chr(27).encode()):
            break
        face_recognize_antrenat(frame, encodings, names)
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    

if __name__=="__main__":
    main()