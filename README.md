# 👁️ Real-Time Face, Eye & Smile Detection Using OpenCV

A Computer Vision project that performs real-time face, eye, and smile detection using OpenCV and Haar Cascade Classifiers. The system captures live webcam video and identifies facial features by drawing bounding boxes around detected faces, eyes, and smiles.

---

## 📌 Project Overview

Computer Vision enables machines to interpret and understand visual information from the real world.

This project uses OpenCV's pre-trained Haar Cascade Classifiers to detect:

- Human Faces
- Eyes
- Smiles

The system processes live webcam frames and performs real-time detection with visual annotations.

---

## 🎯 Objectives

- Detect faces in real-time using a webcam.
- Detect eyes within detected face regions.
- Detect smiles within detected face regions.
- Visualize detections with bounding boxes and labels.
- Learn practical Computer Vision using OpenCV.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Haar Cascade Classifiers

---

## 📂 Project Files

```text
real-time-face-eye-smile-detection/
│
├── face_detect.py
├── haarcascade_frontalface.xml
├── haarcascade_eye.xml
├── haarcascade_smile.xml
├── requirements.txt
├── README.md
└── LICENSE
└── .gitignore
```

---

## 📚 Haar Cascade Classifiers Used

### Face Detection

```text
haarcascade_frontalface_default.xml
```

### Eye Detection

```text
haarcascade_eye.xml
```

### Smile Detection

```text
haarcascade_smile.xml
```

These pre-trained XML files are provided by OpenCV and are used for object detection.

---

## ⚙️ Working Process

### Step 1: Start Webcam

The webcam is activated using:

```python
webcam = cv2.VideoCapture(0)
```

---

### Step 2: Capture Video Frames

The program continuously captures frames from the webcam.

---

### Step 3: Convert to Grayscale

For faster processing:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

### Step 4: Face Detection

Faces are detected using:

```python
face_cascade.detectMultiScale()
```

A green rectangle is drawn around each detected face.

---

### Step 5: Eye Detection

Eyes are detected inside the face region.

A blue rectangle is drawn around each eye.

---

### Step 6: Smile Detection

Smiles are detected inside the face region.

A red rectangle is drawn around each smile.

---

### Step 7: Display Results

The processed video stream is displayed in real time.

---

### Step 8: Exit Program

Press:

```text
q
```

to stop the application.

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/vatsal-dhuvad/real-time-face-eye-smile-detection.git
```

### Navigate to Project Folder

```bash
cd real-time-face-eye-smile-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python face_detect.py
```

---

## 📊 Detection Features

### Face Detection

✔ Detects multiple faces

✔ Draws green bounding boxes

✔ Labels detected faces

---

### Eye Detection

✔ Detects eyes within face region

✔ Draws blue bounding boxes

✔ Labels detected eyes

---

### Smile Detection

✔ Detects smiles in real time

✔ Draws red bounding boxes

✔ Labels smile regions

---

## 📈 Sample Output

```text
Face Detected
Eye Detected
Smile Detected
```

Live webcam output:

```text
[Face]
 ├── [Eye]
 ├── [Eye]
 └── [Smile]
```

---

## 🚀 Future Enhancements

- Face Recognition System
- Attendance System using Face Recognition
- Emotion Detection
- Mask Detection
- Drowsiness Detection
- Gender Classification
- Age Prediction
- Deep Learning Face Detection (DNN)
- YOLO-Based Detection

---

## 💼 Real-World Applications

- Smart Surveillance Systems
- Face Recognition Systems
- Security Authentication
- Attendance Management
- Human-Computer Interaction
- Driver Monitoring Systems
- Emotion Analysis

---

## 📚 Skills Demonstrated

- Computer Vision
- OpenCV
- Image Processing
- Real-Time Video Processing
- Object Detection
- Haar Cascade Classifiers
- Python Programming

---

## 👨‍💻 Author

**Vatsal Dhuvad**

GitHub: https://github.com/vatsal-dhuvad

---

## ⭐ If you found this project useful, consider giving it a star on GitHub.
