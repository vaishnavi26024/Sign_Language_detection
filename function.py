# function.py

# Import dependencies
import cv2
import numpy as np
import os
import mediapipe as mp

# Mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

def extract_keypoints(results):
    hand_landmarks = results.multi_hand_landmarks
    if hand_landmarks:
        all_hands = []
        for hand in hand_landmarks:
            all_hands.append(
                np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
            )
        # If only one hand detected, pad the other with zeros
        if len(all_hands) == 1:
            all_hands.append(np.zeros(21 * 3))
        return np.concatenate(all_hands)
    else:
        # If no hands detected, return zeros for both hands
        return np.zeros(21 * 3 * 2)

# Path for exported data
DATA_PATH = os.path.join('MP_Data')

# Actions
actions = np.array(['A', 'B', 'C','D'])

# Number of sequences per action
no_sequences = 35

# Length of each sequence
sequence_length = 35
