import os
import cv2
import numpy as np


def draw(image_path: str, boxes: np.ndarray, landmarks: np.ndarray = None):
    img = cv2.imread(image_path, 1)
    boxes = boxes.astype(int)

    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    if landmarks is not None:
        landmarks = landmarks.astype(int)
        for landmarks_per_box in landmarks:
            for point in landmarks_per_box:
                cv2.circle(img, point, radius=3, color=(0, 0, 255), thickness=-1)
    save_path = os.path.join("output_images", os.path.split(image_path)[-1])
    if cv2.imwrite(save_path, img):
        print("Saved output image successfully")
    else:
        print("Could not save image")
