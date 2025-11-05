# MULTIMODAL LANGUAGE IDENTIFICATION AND EMOTION RECOGNITION SYSTEM

This project is a Flask-based web application that performs real-time fusion of three modalities—Visual (Webcam), Text (Multilingual Input), and Speech (Live Audio)—to determine a final dominant emotion.

## ✨ Innovative Features

* **Multilingual Text Analysis:** Uses the **Helsinki-NLP/opus-mt-mul-en** model for language detection and translation into English before emotion analysis.
* **Live Audio Recording:** Uses the **RecordRTC** JavaScript library to capture and process microphone audio directly in the browser (requires FFmpeg).
* **Crash-Proof Backend:** Robust `cv2` and `DeepFace` error handling ensures the server remains stable even if the camera fails or no face is detected.
* **Modern Frontend:** Built with **Bootstrap 5** for a responsive, innovative design.

## 🚀 Setup & Installation (3-Step Quick Start)

### 1. Clone Repository & Create Environment

```bash
git clone [Your GitHub Repository URL]
cd [Your Project Folder Name]
python -m venv venv
.\venv\Scripts\activate