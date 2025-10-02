# data.py

from function import *
import cv2
import os
import numpy as np

# Create folders for saving data
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Use Mediapipe Hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                # Read the image for this frame
                frame = cv2.imread('Image/{}/{}.png'.format(action,sequence))

                if frame is None:
                    print(f"[WARNING] Image not found: Image/{action}/{sequence}/{frame_num}.png")
                    continue

                # Process with Mediapipe
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Show status on the first frame
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Extract and save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Exit if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
