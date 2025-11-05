# visual_analysis.py
from deepface import DeepFace
import cv2
import numpy as np

# Initialize the DeepFace model once
try:
    # DeepFace will download models on first run, but initialization is faster
    # than calling analyze every time for large video streams.
    print("Initializing DeepFace model...")
    # Passing a dummy image forces the models to be loaded into memory
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_img, actions=('emotion',), enforce_detection=False, silent=True)
    print("DeepFace model loaded successfully.")
except Exception as e:
    print(f"DeepFace Startup Warning: Could not pre-load model. It will load on first analysis. {e}")

def analyze_visual_emotion(frame):
    """
    Analyzes emotion from a single frame using DeepFace.
    Returns: A list of dicts containing 'dominant_emotion' and 'face_location'.
    """
    if frame is None or frame.size == 0:
        return []

    try:
        # Use BGR2RGB conversion, as DeepFace prefers RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # enforce_detection=False is CRUCIAL to prevent crashes when no face is found
        results = DeepFace.analyze(
            img_path=rgb_frame, 
            actions=('emotion',), 
            enforce_detection=False,
            detector_backend='opencv' # Use OpenCV for fast detection
        )
        
        if not results:
            return []
            
        # DeepFace returns a list of results, we only use the first detected face
        detection = results[0]
        
        # We also return the input frame so app.py can draw the box
        return [{
            'dominant_emotion': detection['dominant_emotion'],
            'face_location': detection['region']
        }]

    except Exception as e:
        # Log the error, but don't crash the server
        print(f"[Visual Analysis Error]: {e}")
        return []