# importing important libraries
import cv2
import numpy as np
import mediapipe as mp
import time
import winsound

#mediaPipe task imports
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions
)
from mediapipe.framework.formats import landmark_pb2

#drawing utils from MediaPipe 
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Path to the face_landmarker.task model
model_path = r"C:\Users\sadai\Downloads\face_landmarker (1).task"

# Configuring without the runningMode for now
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# Create the face landmarker instance
face_landmarker = FaceLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

with face_landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap image for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Use synchronous detection
        result = face_landmarker.detect(mp_image)
        
        #custom faceLandmark style
        landmark_style = mp_drawing.DrawingSpec(color=(200, 0, 255), thickness=1, circle_radius=1)
        connection_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)

        
        #draw face landmark
        if result and result.face_landmarks:
            for face_landmarks in result.face_landmarks:
        # Convert to protobuf-compatible format
                landmark_list = landmark_pb2.NormalizedLandmarkList()
            for lm in face_landmarks:
                landmark_list.landmark.append(
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            )        
        #drawing the mesh
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = landmark_list,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = landmark_style,
                connection_drawing_spec = connection_style     
                )
        # Get the blendshapes and print eye blinks
        if result and result.face_blendshapes:
            for blendshape in result.face_blendshapes[0]:
                if blendshape.category_name in ['eyeBlinkLeft', 'eyeBlinkRight']:
                    print(f"{blendshape.category_name}: {blendshape.score: }")


        # Display the webcam feed
        cv2.imshow('MediaPipe FaceLandmarker', frame)

        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows() 
