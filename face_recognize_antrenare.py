"""
Usage:
  face_recognize_antrenare.py -d <train_dir>
  
Options:
  -h, --help                     Show this help
  -d, --train_dir =<train_dir>   Directory with 
                                 images for training
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

def face_recognize(dir):
    # Training the SVC classifier
    # The training data would be all the 
    # face encodings from all the known 
    # images and the labels are their names
    encodings = []
    names = []
    encodings.clear
    names.clear

    # Training directory
    if dir[-1]!='/':
        dir += '/'
    train_dir = os.listdir(dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(dir + person)
  
        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                dir + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)
  
            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image 
                # with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " can't be used for training")

    # Write the encodings and names in separate files
    joblib.dump(encodings, "encodings.bin")
    joblib.dump(names, "names.bin")

def main():
    args = docopt.docopt(__doc__)
    train_dir = args["--train_dir"]
    face_recognize(train_dir)
  
if __name__=="__main__":
    main()