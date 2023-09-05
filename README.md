# Facial_Recognition
# Firstly, make sure that the file structure for the training data (images) follows this template:
train_dir/
    person_1/
        person_1_face-1.jpg
        person_1_face-2.jpg
        .
        .
        person_1_face-n.jpg
    person_2/
        person_2_face-1.jpg
        person_2_face-2.jpg
        .
        .
        person_2_face-n.jpg
    .
    .
    person_n/
        person_n_face-1.jpg
        person_n_face-2.jpg
        .
        .
        person_n_face-n.jpg
# Then, run the face_recognize_antrenare.py file. This will create encodings.bin and names.bin binary files, thus trainig the AI model
# Lastly, run the face_recognize_video_final.py file in order to get live video facial recognition
