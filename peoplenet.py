import cv2
import numpy as np
import onnxruntime as ort

from utils import nms


class PeopleNet:
    def __init__(self, onnx_path="models/resnet34_peoplenet_int8.onnx"):
        self.stride = 16
        self.box_norm = 35
        self.input_size = [544, 960]

        self.n_classes = 3

        self.anchors = self._generate_anchors()

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def _generate_anchors(self):
        grid_height = int(self.input_size[0] / self.stride)
        grid_width = int(self.input_size[1] / self.stride)
        y_coords, x_coords = np.meshgrid(
            np.arange(grid_height), np.arange(grid_width), indexing="ij"
        )
        anchor_centers = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)
        anchor_centers = anchor_centers * self.stride / self.box_norm
        anchor_centers = anchor_centers.repeat(self.n_classes, axis=0)
        anchor_centers = anchor_centers.astype(np.float32)
        return anchor_centers

    def _forward(self, img: np.ndarray):
        def postprocess(out: np.ndarray):
            out = out[0].transpose(1, 2, 0)
            out = out.reshape(out.shape[0] * out.shape[1] * self.n_classes, -1)
            return out

        scores, boxes = self.session.run(None, {self.input_name: img})
        return postprocess(scores), postprocess(boxes)

    def detect(self, image_path, conf_thres=0.5, nms_thres=0.4):
        img = cv2.imread(image_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        aspect_ratio = w / h

        if aspect_ratio > (self.input_size[1] / self.input_size[0]):
            new_w = self.input_size[1]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = self.input_size[0]
            new_w = int(new_h * aspect_ratio)

        img = cv2.resize(img, (new_w, new_h))
        img = np.pad(
            img,
            ((0, self.input_size[0] - new_h), (0, self.input_size[1] - new_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = img / 255

        scores, boxes = self._forward(img)

        scaling_factor = new_h / h
        centers = self.anchors - boxes[:, :2]
        wh = self.anchors + boxes[:, 2:]
        boxes = np.hstack([centers, wh]) * self.box_norm

        ids = np.where(scores > conf_thres)[0]
        scores = scores[ids]
        boxes = boxes[ids]

        order = scores.argsort(0)[::-1]
        keep = nms(boxes, order[:, 0], nms_thres)
        scores = scores[keep]
        boxes = boxes[keep]
        boxes /= scaling_factor

        return np.hstack([scores, boxes])


if __name__ == "__main__":
    from utils import draw

    image_path = "images/solvay_conference_1927.jpg"
    model = PeopleNet()
    dets = model.detect(image_path, 0.4)
    print("Num detects:", dets.shape[0])
    draw(image_path, dets[:, 1:])
