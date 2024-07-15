import cv2
import os
import mediapipe as mp
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

dp = 'C:\\Users\\MADHU MITHRA\\Downloads\\sample'
data = []
labels = []
c = 0

for i in os.listdir(dp):
    if i == 'desktop.ini':
        continue
    
    print(f"Processing folder: {i}")
    folder_path = os.path.join(dp, i)
    folder_count = 0  # Count processed images per folder
    
    for j in os.listdir(folder_path):
        data_aux = []
        img_path = os.path.join(folder_path, j)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Failed to read image: {img_path}")
            continue  # Skip if image cannot be read
        
        img_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_c)
        print(f"Processing image: {img_path}")
        
        if results.multi_hand_landmarks:
            print(f"Hand landmarks found in image: {img_path}")
            c += 1
            folder_count += 1
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    data_aux.append(x)
                    data_aux.append(y)
            
            data.append(data_aux)
            labels.append(i)
        else:
            print(f"No hand landmarks found in image: {img_path}")
    
    print(f"Total images processed in folder {i}: {folder_count}")

print(f"Total images processed: {len(data)}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
