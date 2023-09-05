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
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    no = len(face_locations)
    cv2.putText(test ,"Number of faces detected: " + str(no), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 0), 2)
    
    # Find all the faces and face encodings in the current frame of video
    face_names = []
    for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]
            face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(test, (left, top), (right, bottom), (255, 69, 0), 2)
        # Draw a label with a name below the face
        cv2.putText(test, str(name), (left + 6, bottom - 6), cv2.FONT_HERSHEY_PLAIN, 1.0, (127, 255, 0), 2)

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