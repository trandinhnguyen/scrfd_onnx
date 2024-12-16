import sys
import os
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage
from scrfd import SCRFD


def draw_faces(
    image: PILImage,
    boxes: np.ndarray,
    kpss: np.ndarray,
    radius: int = 3,
    box_width: int = 3,
) -> PILImage:
    draw = ImageDraw.Draw(image)

    for box, keypoints in zip(boxes, kpss):
        draw.rectangle([int(e) for e in box], outline="red", width=box_width)
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            ellipse = [(x - radius, y - radius), (x + radius, y + radius)]
            draw.ellipse(ellipse, fill="red")

    return image


image_path = "images/worlds_largest_selfie_image.jpg"
head, tail = os.path.split(image_path)
image = Image.open(image_path).convert("RGB")
model_path = "./models/scrfd.onnx"

face_detector = SCRFD(model_path)
boxes, kpss, scores = face_detector.detect(image, 0.3, 0.4)
result = draw_faces(image, boxes, kpss)
save_path = os.path.join("output_images", tail)
result.save(save_path)
