# 📸 Real-Time Face Detection with Facial Landmarks

A modern, high-speed face detection application built with **Python**, **OpenCV**, and **Google MediaPipe (2025)**. This tool detects faces and 6 key landmarks (eyes, nose, mouth, and ears) in real-time using a deep learning model.

---

## ✨ Features
- **Real-Time Detection:** Smooth 60FPS detection on most modern CPUs.
- **Deep Learning Based:** Uses the BlazeFace (TFLite) architecture for high precision.
- **Facial Landmarks:** Accurately plots 6 essential points on the face.
- **Safe Exit:** Handles `KeyboardInterrupt` (Ctrl+C) gracefully to prevent system crashes.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone github.com
   cd your-repo-name

2. **Install dependencies:**
```bash
pip install opencv-python mediapipe
```
3. **Download the Model:**
Download blaze_face_short_range.tflite and place it in the project root folder.

## 🚀 Usage
Simply run the script to start your webcam:
```bash
python webcam_detect.py
```
- **Press 'q' to quit the window.**
- **Press 'Ctrl+C' in the terminal to stop the process.**
## 📝 How it Works
1. **Model Selection: Uses MediaPipe Tasks API for 2025.**
2. **Color Conversion: Transforms BGR (OpenCV default) to RGB (MediaPipe requirement).**
3. **Keypoint Mapping: Maps 6 specific points:**

**1.  👁️ Right & Left Eyes**
**2. 👃 Nose Tip**
**3. 👄 Mouth Center**
**4. 👂 Right & Left Ear Tragus**
