# Facial Recognition System

A Python-based face recognition system using the `face_recognition` library with support for training on static images, testing on individual images, and real-time video recognition.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Training Data Structure](#training-data-structure)
- [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Webcam (for video recognition)
- Linux/macOS/Windows system

### Virtual Environment Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/Facial_Recognition
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   
   **On Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```
   
   **On Windows:**
   ```bash
   .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following packages (specified in `requirements.txt`):

```
face_recognition>=1.3.0
face_recognition_models>=0.3.0
scikit-learn>=0.22.0
opencv-python>=4.5.0
numpy>=1.19.0
joblib>=1.0.0
docopt>=0.6.2
setuptools<75
```

**Note:** If you encounter issues with `face_recognition_models`, try installing from GitHub:
```bash
pip install git+https://github.com/ageitgey/face_recognition_models
```

## Usage

### Running the Application

The application uses an interactive command-line interface. Run the main script:

```bash
python main.py
```

### Available Commands

Once running, you'll see the interactive prompt with these commands:

- **`train <train_dir>`** - Train the face recognition model with images from a directory
- **`test <test_image>`** - Test recognition on a static image
- **`video`** - Run real-time video face recognition using webcam
- **`status`** - Check if trained model files exist
- **`help`** - Show available commands
- **`exit`** or **`quit`** - Exit the program

### Workflow

1. **Prepare training data** (see [Training Data Structure](#training-data-structure))
2. **Train the model:**
   ```
   > train train_dir/
   ```
   This creates `encodings.bin` and `names.bin` files in the project root.
3. **Test on static image:**
   ```
   > test path/to/test_image.jpg
   ```
4. **Run video recognition:**
   ```
   > video
   ```
   Press `ESC` to exit the video feed.

### Example Session

```bash
$ python main.py

==================================================
Facial Recognition Hub - Interactive Mode
==================================================

Model Status:  No trained model found

Available commands:
  train <train_dir>    - Train model with images from directory
  test <test_image>    - Test recognition on a static image
  video                - Run real-time video face recognition
  status               - Check model status
  h/help               - Show this help message
  q/exit/quit          - Exit the program

Example:
  > train training_data/
  > test test_image.jpg
  > video
==================================================

> train train_dir/
Training model with images from: train_dir/
Training complete! Encodings saved to encodings.bin and names.bin

> status

Model Status:  ✓ Trained model found (encodings.bin + names.bin)

> test test_photo.jpg
Number of faces detected:  1
Found:
John_Doe

> video
Starting video face recognition...
Press 'ESC' to exit
[Video feed runs until ESC is pressed]

> exit
Exiting...
```

## Training Data Structure

Organize your training images in the following structure:

```
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
```

**Important notes:**
- Each person should have their own subdirectory named after them
- Include multiple images per person for better accuracy
- Each training image should contain exactly one face
- Images with multiple faces or no faces will be skipped during training
- Supported formats: JPG, PNG, and other image formats supported by PIL

## Implementation Details

### Architecture

The project is organized into three main modules:

1. **`src/train_encodings.py`** - Training module
   - Function: `train_and_save_encodings(train_dir)`
   - Processes training images and generates face encodings
   - Saves encodings and names to binary files

2. **`src/recognize_static_image.py`** - Static image testing module
   - Function: `recognize_faces_in_image(test_image_path)`
   - Loads pre-trained encodings
   - Tests recognition on a single image using SVM classifier

3. **`src/recognize_on_video_feed.py`** - Video recognition module
   - Function: `run_video_recognition()`
   - Function: `recognize_faces_in_frame(frame, encodings, names)`
   - Performs real-time face recognition on webcam feed
   - Uses face comparison method (not SVM)

### Technical Details

- **Face Detection**: Uses HOG-based model by default (CNN available but slower)
- **Face Encoding**: 128-dimensional face embeddings
- **Classification**: 
  - Static images: SVM classifier with `gamma='scale'`
  - Video: Direct face comparison using Euclidean distance
- **Model Persistence**: Uses `joblib` for serializing encodings and names
- **Path Handling**: All modules use `PROJECT_ROOT` for consistent file path resolution

### Key Functions

#### train_and_save_encodings(train_dir)
```python
# Processes training images and saves encodings
# Input: Path to training directory
# Output: Creates encodings.bin and names.bin
```

#### recognize_faces_in_image(test_image_path)
```python
# Tests recognition on a static image
# Input: Path to test image
# Output: Prints detected faces and their names
```

#### run_video_recognition()
```python
# Runs real-time video recognition
# Input: None (uses webcam)
# Output: Video feed with face annotations
```

## Project Structure

```
Facial_Recognition/
├── main.py                      # Interactive CLI entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── TODO.md                      # Project TODO list
├── .venv/                       # Virtual environment (created during setup)
├── src/                         # Source modules
│   ├── __init__.py             # Package initialization
│   ├── train_encodings.py      # Training module
│   ├── recognize_static_image.py  # Static image testing
│   └── recognize_on_video_feed.py  # Video recognition
├── deprecated/                  # Deprecated scripts (kept for reference)
│   ├── face_recognize.py
│   ├── face_recognize_antrenat.py
│   └── face_recognize_video_prost.py
├── train_dir/                   # Training data directory (example)
│   ├── First_Person/
│   └── Other_Person/
├── encodings.bin               # Generated face encodings (after training)
└── names.bin                   # Generated person names (after training)
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'face_recognition'**
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`

**2. Qt platform plugin warnings (Linux)**
- These warnings don't affect functionality
- To silence: `export QT_QPA_PLATFORM=xcb` before running
- Or install: `sudo apt install libxcb-cursor0`

**3. "The number of classes has to be greater than one"**
- Training data must contain at least 2 different people
- Add more people to your training directory

**4. Webcam not accessible**
- Check webcam permissions
- Ensure no other application is using the webcam
- Try changing camera index in `recognize_on_video_feed.py` (line 56)

**5. Face detection not working**
- Ensure images have good lighting
- Faces should be clearly visible and frontal
- Try higher resolution images

**6. Import errors after restructuring**
- Ensure you're running from project root directory
- Virtual environment should be activated
- Check that `src/__init__.py` exists

### Performance Tips

- **Training speed**: Use fewer images per person for faster training
- **Recognition accuracy**: Use 5-10 images per person for better accuracy
- **Video performance**: Reduce resolution if video is laggy
- **Memory usage**: Clear old encodings.bin files before retraining
