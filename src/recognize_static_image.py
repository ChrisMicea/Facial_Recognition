"""
Testing module for face recognition on static images.
Loads pre-trained encodings and tests recognition on a single image.
"""

# importing libraries
import face_recognition
from sklearn import svm
import joblib
import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def recognize_faces_in_image(test_image_path):
    encodings = []
    names = []
    #read encodings from file
    encodings_path = os.path.join(PROJECT_ROOT, "encodings.bin")
    names_path = os.path.join(PROJECT_ROOT, "names.bin")
    encodings = joblib.load(encodings_path)
    names = joblib.load(names_path)
    # Create and train the SVC classifier
    clf = svm.SVC(gamma ='scale')
    clf.fit(encodings, names)

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(test_image_path)
  
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