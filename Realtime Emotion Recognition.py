import sys
import cv2
import numpy as np
from keras.models import load_model

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load pre-trained emotion recognition model
model = load_model('C:/Users/mahes/Rvu-Python/Identify emotions/model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y + h, x:x + w]
        face = preprocess_face(face)
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion, (x, y, w, h)
    return None, None

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        emotion, face_coords = predict_emotion(frame)
        if emotion and face_coords:
            x, y, w, h = face_coords
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    except Exception as e:
        print(f"Error: {e}")
    
    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
