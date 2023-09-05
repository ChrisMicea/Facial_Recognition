"""
Usage:
  face_recognize_run.py -i <test_image>
  
Options:
  -h, --help                     Show this help
  -i, --test_image =<test_image> Test image
"""
  
# importing libraries
import face_recognition
import sys
import docopt
from sklearn import svm
import os
from array import array
import numpy as np
import joblib

def face_recognize_antrenat(test):
    encodings = []
    names = []
    #read encodings from file
    encodings = joblib.load("encodings.bin")
    names = joblib.load("names.bin")
    # Create and train the SVC classifier
    clf = svm.SVC(gamma ='scale')
    clf.fit(encodings, names)

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(test)
  
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)
  
    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        print(*name)

def main():
    args = docopt.docopt(__doc__)
    test_image = args["--test_image"]
    face_recognize_antrenat(test_image)
  
if __name__=="__main__":
    main()