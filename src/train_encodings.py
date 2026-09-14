"""
Training module for face recognition.
Generates face encodings from training images and saves them to binary files.
"""

# importing libraries
import face_recognition
import os
import joblib

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_and_save_encodings(train_dir):
    # Training the SVC classifier
    # The training data would be all the 
    # face encodings from all the known 
    # images and the labels are their names
    encodings = []
    names = []
    encodings.clear
    names.clear

    # Training directory
    if train_dir[-1]!='/':
        train_dir += '/'
    train_dir_list = os.listdir(train_dir)

    # Loop through each person in the training directory
    for person in train_dir_list:
        pix = os.listdir(train_dir + person)
  
        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                train_dir + person + "/" + person_img)
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
    encodings_path = os.path.join(PROJECT_ROOT, "encodings.bin")
    names_path = os.path.join(PROJECT_ROOT, "names.bin")
    joblib.dump(encodings, encodings_path)
    joblib.dump(names, names_path)