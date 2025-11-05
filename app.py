# app.py
import os
import cv2
import numpy as np
import base64
import time
import librosa
import tempfile
from collections import Counter
from pydub import AudioSegment

from flask import Flask, render_template, Response, request, jsonify

# Import analysis modules
# NOTE: These MUST be loaded before Flask starts serving
try:
    from visual_analysis import analyze_visual_emotion 
    from language_tools import identify_and_translate_text 
    print("All custom analysis modules successfully imported.")
except Exception as e:
    print(f"CRITICAL ERROR during module import. Check requirements.txt and file names: {e}")
    exit()

# --- CAMERA INITIALIZATION (Will try index 0, then 1, then fail gracefully) ---
def init_camera():
    for index in range(3):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam successfully opened at index {index}.")
            return cap
        cap.release()
    print("CRITICAL: Could not open any webcam index (0, 1, or 2).")
    return cv2.VideoCapture(-1) # Return a failed capture object

app = Flask(__name__)
camera = init_camera()
latest_visual_emotion = "Neutral" 

# --- HELPER FUNCTIONS ---

def multimodal_fusion(visual_emotion, speech_emotion, text_emotion):
    """Combines emotion results to find a single dominant emotion."""
    emotions = [visual_emotion, speech_emotion, text_emotion]
    # Filter out placeholder/error messages/Neutral for better fusion result
    valid_emotions = [e for e in emotions if e.lower() not in ('n/a', 'no data', 'neutral', 'calm') and not e.startswith(('Audio Error', 'Translation Failed'))]
    
    if not valid_emotions:
        return "Neutral" if 'Neutral' in emotions else "No Input"
        
    counts = Counter(valid_emotions)
    return counts.most_common(1)[0][0]

def generate_frames():
    """Generates frames from the webcam for the web browser."""
    global latest_visual_emotion
    
    if not camera.isOpened():
        # Display a "Camera Not Found" placeholder image if the camera failed to open
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "CAMERA FAILED TO LOAD", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return # Stop frame generation

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(1) 
            continue

        try:
            # 1. VISUAL EMOTION ANALYSIS
            detections = analyze_visual_emotion(frame)
            
            if detections:
                result = detections[0]
                dominant_emotion = result.get('dominant_emotion', 'N/A').capitalize()
                latest_visual_emotion = dominant_emotion # Update global state
                
                # Draw results on frame
                face_location = result.get('face_location')
                if face_location:
                    x = face_location['x']
                    y = face_location['y']
                    w = face_location['w']
                    h = face_location['h']

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, 
                        f"Visual: {dominant_emotion}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2
                    )
            else:
                 latest_visual_emotion = "Neutral" # Reset if no face is detected

        except Exception as e:
            print(f"Error drawing or running visual analysis: {e}")
            latest_visual_emotion = "Analysis Error"
            
        # 2. Encode frame and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream the webcam video with emotion analysis."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_visual_emotion')
def get_visual_emotion():
    """Returns the latest visual emotion for dynamic frontend update."""
    global latest_visual_emotion
    return jsonify({'visual_emotion': latest_visual_emotion})

@app.route('/process_text_audio', methods=['POST'])
def process_text_audio():
    """Handles text and audio input submission and runs fusion."""
    global latest_visual_emotion
    
    try:
        # --- 1. TEXT INPUT ---
        text_input = request.form.get('text_input', '')
        translated_text, lang_status, text_emotion = "No Input", "N/A", "No Data"

        if text_input.strip():
            translated_text, lang_status = identify_and_translate_text(text_input)
            
            # Simple Text Emotion Placeholder (Replace with your actual Text Emotion Model)
            if "happy" in translated_text.lower() or "joy" in translated_text.lower():
                text_emotion = "Happy"
            elif "sad" in translated_text.lower() or "grief" in translated_text.lower():
                text_emotion = "Sad"
            elif "angry" in translated_text.lower() or "rage" in translated_text.lower():
                text_emotion = "Angry"
            else:
                text_emotion = "Neutral"
        
        # --- 2. AUDIO INPUT ---
        speech_emotion = "No Data"
        audio_file = request.files.get('audio_file')
        
        if audio_file and audio_file.filename != '':
            temp_path = None
            try:
                # Use tempfile to securely handle and clean up the uploaded file
                suffix = os.path.splitext(audio_file.filename)[1]
                # If audio is recorded, it's often a webm blob, which needs conversion
                if suffix.lower() == '.webm':
                     # We save the webm and convert it to a temp WAV for librosa
                    temp_webm_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                    audio_file.save(temp_webm_path)
                    
                    audio = AudioSegment.from_file(temp_webm_path, "webm")
                    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                    audio.export(temp_path, format="wav")
                    os.remove(temp_webm_path)
                else:
                    # Handle normal file upload
                    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
                    audio_file.save(temp_path)

                
                # Load audio using librosa
                y, sr = librosa.load(temp_path, sr=None)
                
                # Simple Audio Emotion Placeholder (Replace with your Speech Emotion Model)
                duration = librosa.get_duration(y=y, sr=sr)
                if duration > 5:
                     speech_emotion = "Surprise"
                elif duration < 2:
                     speech_emotion = "Fear"
                else:
                     speech_emotion = "Calm"

            except Exception as e:
                speech_emotion = f"Audio Error: Check FFmpeg/Dependencies. {str(e)}"
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        
        # --- 3. FUSION ---
        # Use the latest emotion detected from the camera stream
        final_emotion = multimodal_fusion(latest_visual_emotion, speech_emotion, text_emotion)
        
        # 4. Return Results
        return jsonify({
            'translated_text': translated_text,
            'lang_status': lang_status,
            'text_emotion': text_emotion,
            'speech_emotion': speech_emotion,
            # Return the latest camera detected emotion
            'visual_emotion': latest_visual_emotion, 
            'final_emotion': final_emotion
        })
        
    except Exception as e:
        print(f"\n[FATAL ERROR in process_text_audio]: {e}\n")
        # Return 500 status to alert frontend
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

# --- APP EXECUTION ---

if __name__ == '__main__':
    # Ensure camera is released if it was previously open
    if camera.isOpened():
        # Since init_camera might have returned a failed capture, check again
        pass 
    
    print("\n--- Starting Flask Web App ---")
    print("Open your browser to: http://127.0.0.1:5000/")
    
    # use_reloader=False is key to preventing re-import errors
    app.run(debug=True, use_reloader=False)