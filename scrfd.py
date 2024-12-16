import numpy as np
from onnxruntime import InferenceSession
from PIL import Image
from PIL.Image import Image as PILImage


def normalize(
    img: np.ndarray,
    scaling: float = 1.0 / 127.5,
    mean: tuple[float, float, float] = (127.5, 127.5, 127.5),
) -> np.ndarray:
    """Change dtype, normalize, and transpose."""
    img = img.astype(dtype=np.float32)
    img -= mean
    img *= scaling
    img = img.transpose(2, 0, 1)
    return img


def distance2bbox(
    points: np.ndarray, distance: np.ndarray, max_shape: tuple | None = None
) -> np.ndarray:
    """Compute exact bounding boxes coordinates for one feature level by using
    anchor centers and relative distances between boxes and anchors.

    Args:
        - points: Anchor centers.
        - distance: Box offsets.
        - max_shape: The final box coordinates will be clipped to these values.

    Returns:
        - The final box coordinates with shape of (:, 4).
    """
    x1 = points[:, 0] - distance[:, 0]  # shape (:,)
    y1 = points[:, 1] - distance[:, 1]  # shape (:,)
    x2 = points[:, 0] + distance[:, 2]  # shape (:,)
    y2 = points[:, 1] + distance[:, 3]  # shape (:,)

    if max_shape:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])

    return np.stack([x1, y1, x2, y2], axis=-1)  # shape (:, 4)


def distance2kps(
    points: np.ndarray, distance: np.ndarray, max_shape: tuple | None = None
) -> np.ndarray:
    """Compute exact keypoints coordinates for one feature level by using
    anchor centers and relative distances between keypoints and anchors.

    Args:
        - points: Anchor centers.
        - distance: Keypoints offsets.
        - max_shape: The final keypoints coordinates will be clipped to these values.

    Returns:
        - The final keypoints coordinates with shape of (:, 5, 2).
    """
    preds = []
    bound = distance.shape[1]  # = 10
    for i in range(0, bound, 2):
        px = points[:, i % 2] + distance[:, i]  # shape (:,)
        py = points[:, i % 2 + 1] + distance[:, i + 1]  # shape (:,)

        if max_shape:
            px = px.clip(min=0, max=max_shape[1])
            py = py.clip(min=0, max=max_shape[0])

        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape(distance.shape[0], -1, 2)


def nms(boxes: np.ndarray, order: np.ndarray, nms_thresh: float) -> list:
    """Perform non-maximum suppression on bounding boxes

    Args:
        - boxes: bounding boxes have shape (:, 4)
        - order: sorted indices
        - nms_threshold: iou threshold
    Returns:
        - list of indices"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)  # always keep the first element

        # Compute ious
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= nms_thresh)[0]  # get rid of boxes whose iou is large
        order = order[inds + 1]  # increase indices by 1 to eliminate current box
    return keep


class SCRFD:
    def __init__(self, model_path):
        """Define some model configurations and initialize child objects."""
        # Model configs
        self.input_shape = [640, 640, 3]
        self.num_anchors = 2
        self.feat_stride_fpn = [8, 16, 32]
        self.n_feature_maps = len(self.feat_stride_fpn)

        # Anchors for all feature levels in pyramid.
        self.anchors = self._generate_anchors()

        # Prepare onnx model and inputs/outputs metadata
        self._load_model(model_path)

    def _load_model(self, model_path):
        self.session = InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]

        input = self.session.get_inputs()[0]
        self.input_name = input.name

        print("Warming up ...")
        self.forward(np.zeros(self.input_shape), 1)

    def _generate_anchors(self):
        """Generate anchors for all feature levels in feature pyramid.

        Returns:
            - Center coordinate of anchors.
        """
        anchors = []
        for stride in self.feat_stride_fpn:
            feature_height = self.input_shape[0] // stride
            feature_width = self.input_shape[1] // stride

            anchor_centers = np.mgrid[:feature_height, :feature_width]  # 0 to h, 0 to w
            anchor_centers = anchor_centers[::-1]  # 0 to w, 0 to h
            anchor_centers = anchor_centers.transpose(1, 2, 0)  # shape (h, w, 2)
            anchor_centers = anchor_centers.reshape(-1, 2)  # shape (h*w, 2)
            anchor_centers *= stride

            if self.num_anchors > 1:
                """
                NOTE: Both np.concatenate and the line below return an array of
                shape (:, 4). However, np.concatenate(..., axis=0) returns array
                like [0, ..., n, 0, ..., n] and we need array like
                [0, 0, 1, 1, ..., n, n], so we use np.concatenate(..., axis=1)
                and then reshape it instead.
                """
                anchor_centers = np.hstack([anchor_centers] * 2).reshape((-1, 2))

            anchors.append(anchor_centers.astype(np.float32))
        return anchors

    def forward(
        self, image: np.ndarray, scores_thresh: float
    ) -> tuple[list, list, list]:
        """Run model inference. Takes resized image and classification threshold
        as inputs, and returns a tuple containing a list of class probabilities,
        a list of bounding boxes, and a list of keypoint coordinates.

        Args:
            - image: resized image as np.ndarray
            - scores_threshold: probability threshold.

        Return:
            - tuple containing scores_list, bboxes_list, and kpss_list
        """
        # Define some variables
        scores_list = []
        bboxes_list = []
        kpss_list = []

        # Preprocess image
        image = normalize(image)
        image = np.expand_dims(image, axis=0)

        # Inference
        net_outs = self.session.run(self.output_names, {self.input_name: image})

        # Loop through feature level in feature pyramid
        for idx, stride in enumerate(self.feat_stride_fpn):
            # Get outputs of each feature level in network's outputs
            scores = net_outs[idx][0]  # shape (:, 1)
            bbox_preds = net_outs[idx + self.n_feature_maps][0] * stride
            kps_preds = net_outs[idx + self.n_feature_maps * 2][0] * stride
            anchor_centers = self.anchors[idx]

            # boxes have shape (:, 4) and kpss have shape (:, 5, 2)
            bboxes = distance2bbox(anchor_centers, bbox_preds, self.input_shape[:2])
            kpss = distance2kps(anchor_centers, kps_preds, self.input_shape[:2])

            pos_inds = np.where(scores >= scores_thresh)[0]
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list  # These arrays have length of 3

    def detect(
        self,
        image: PILImage,
        threshold: float | None = 0.4,
        iou_threshold: float | None = 0.5,
        max_faces: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect faces in image.

        Processing steps:
        1. Resize image without losing aspect ratio and pad asymmetrically
        2. Run inference
        3. Scale predictions back so that boxes, keypoints can be drawn
        properly on the original image.
        4. Apply NMS
        5. (Optional) Limit return predictions

        Args:
            - image: PIL Image
            - threshold: probability threshold
            - iou_threshold: If two boxes have iou larger than this value, one
            will be get rid of
            - max_faces: The number of highest confidence faces will be kept

        Returns:
            - boxes (:, 4), keypoints (:, 5, 2), scores: (:, 1)
        """
        # Resize image without losing aspect ratio and convert to ndarray.
        image = image if image.mode == "RGB" else image.convert("RGB")
        aspect_ratio = image.height / image.width
        if aspect_ratio > 1.0:
            new_height = self.input_shape[0]
            new_width = int(new_height / aspect_ratio)
        else:
            new_width = self.input_shape[1]
            new_height = int(new_width * aspect_ratio)

        resized_img = image.resize((new_width, new_height), resample=Image.NEAREST)
        resized_img = np.asarray(resized_img)

        # Pad image asymmetrically
        padded_img = np.zeros(tuple(self.input_shape), dtype=np.uint8)
        padded_img[:new_height, :new_width, :] = resized_img

        # Scale image and run inference
        scores_list, bboxes_list, kpss_list = self.forward(padded_img, threshold)

        if len(scores_list) == 0:
            print("No faces found!")
            return None

        # Scale back to be compatible with original image
        scaling_factor = new_height / image.height
        bboxes = np.vstack(bboxes_list) / scaling_factor
        kpss = np.vstack(kpss_list) / scaling_factor

        # Get sorted indices by probability
        scores = np.vstack(scores_list)  # shape (:, 1)
        scores_ravel = scores.ravel()  # shape (:,)
        order = scores_ravel.argsort()[::-1]

        # Apply NMS
        keep = nms(bboxes, order, iou_threshold)
        scores = scores[keep, :]
        bboxes = bboxes[keep, :]
        kpss = kpss[keep, :, :]

        # Get faces have highest confidence
        if max_faces and bboxes.shape[0] > max_faces:
            scores = scores[:max_faces]
            bboxes = bboxes[:max_faces]
            kpss = kpss[:max_faces]

        return bboxes, kpss, scores
