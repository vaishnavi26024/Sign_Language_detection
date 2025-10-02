from function import *
from keras.models import model_from_json
import cv2
import numpy as np
import pyttsx3

# Load model
with open("model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Ensure actions are defined
# Example: actions = ['A', 'B', 'C', 'D', 'E']
# You should already import it from function.py if defined there

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Colors for probability bar (optional)
colors = [(245,117,16) for _ in range(20)]

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# Detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8
last_spoken = ""
frame_count = 0
prediction_interval = 3  # Predict every 3rd frame

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0,40), (300,400), 255, 2)

        image, results = mediapipe_detection(cropframe, hands)
        keypoints = extract_keypoints(results)

        # Only collect and predict every few frames
        if frame_count % prediction_interval == 0:
            sequence.append(keypoints)
            sequence = sequence[-35:]

            try:
                if len(sequence) == 35:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    prediction_idx = np.argmax(res)
                    prediction_label = actions[prediction_idx]
                    prediction_conf = res[prediction_idx]

                    print(f"Prediction: {prediction_label} | Confidence: {prediction_conf:.2f}")
                    predictions.append(prediction_idx)
                    predictions = predictions[-10:]

                    # Stable prediction & confident
                    if predictions.count(prediction_idx) > 7 and prediction_conf > threshold:
                        if not sentence or prediction_label != sentence[-1]:
                            sentence.append(prediction_label)
                            accuracy.append(str(round(prediction_conf * 100, 2)))

                            # Speak only if new
                            if prediction_label != last_spoken:
                                engine.say(prediction_label)
                                engine.runAndWait()
                                last_spoken = prediction_label

                            # 🔁 Reset after prediction
                            sequence = []
                            predictions = []

                # Keep only last result
                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

            except Exception as e:
                # For debugging
                # print(e)
                pass

        # Show output
        cv2.rectangle(frame, (0,0), (400, 40), (245, 117, 16), -1)
        output_text = f"Output: {''.join(sentence)} | {''.join(accuracy)}%"
        cv2.putText(frame, output_text, (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # Optional: Visualize prediction bar
        # if len(sequence) == 35 and 'res' in locals():
        #     frame = prob_viz(res, actions, frame, colors, threshold)

        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
