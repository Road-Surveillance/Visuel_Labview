import numpy as np
import torch
import os
import cv2
import json
import shutil
from typing import List
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath("../yolov5"))

from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load

class Detection:
    def __init__(self, weights_path=".pt", size=(640, 640), device="cpu", iou_thres=0.5, conf_thres=0.1):
        self.device = device
        self.model, self.names = self.load_model(weights_path)
        self.size = size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

    def detect(self, frame):
        results, resized_img = self.yolo_detection(frame)
        return results, resized_img

    def preprocess_image(self, original_image):
        resized_img = self.ResizeImg(original_image, size=self.size)
        cv2.imwrite("imageResized.jpg", resized_img)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.float() / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img

    def yolo_detection(self, image, classes=None, agnostic_nms=True, max_det=1000):
        img, resized_img = self.preprocess_image(image.copy())
        pred = self.model(img, augment=False)[0]
        detections = non_max_suppression(
            pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=classes,
            agnostic=agnostic_nms, multi_label=True, labels=(), max_det=max_det
        )
        results = []
        for det in detections:
            det = det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    result = [
                        self.names[int(cls)],
                        str(conf),
                        (xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    ]
                    results.append(result)
        return results, resized_img

    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        h, w = size
        if w1 < h1 * (w / h):
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - int(float(w1 / h1) * h), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img

    def load_model(self, path, train=False):
        model = attempt_load(path, map_location=self.device)
        names = model.module.names if hasattr(model, "module") else model.names
        model.train() if train else model.eval()
        return model, names

    def xyxytoxywh(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = (x[0] + x[2]) / 2  # center x
        y[1] = (x[1] + x[3]) / 2  # center y
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
        return y

def detect_and_export_json(image_path, obj_detector):
    image = cv2.imread(image_path)
    if image is None:
        return json.dumps({"error": f"Could not load image: {image_path}"})

    detections, resized_img = obj_detector.detect(image.copy())
    output_img = resized_img.copy()

    results_list = []  # List to store detection details

    for idx, det in enumerate(detections):
        label, conf, box = det
        if label.lower() == "person":
            continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            output_img,
            f"{label} {float(conf):.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

        crop_img = resized_img[y1:y2, x1:x2].copy()
        if label.lower() in ["square license plate", "rectangle license plate"]:
            crop_folder = "LPs"
            det_type = "license_plate"
        else:
            crop_folder = "Vehicle"
            det_type = "vehicle"

        crop_filename = os.path.join(
            crop_folder,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_{det_type}_{idx}.jpg",
        )
        cv2.imwrite(crop_filename, crop_img)
        results_list.append(
            {
                "label": label,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "crop_path": crop_filename,
                "detection_type": det_type,
            }
        )

    annotated_path = os.path.join("out", os.path.basename(image_path))
    cv2.imwrite(annotated_path, output_img)

    result_dict = {"annotated_image": annotated_path, "detections": results_list}
    return json.dumps(result_dict, indent=2)

# Main function for running under LabVIEW
def process_image(image_path):
    obj_detector = Detection(weights_path="object.pt", device="cpu")
    json_output = detect_and_export_json(image_path, obj_detector)
    return json_output


json_output=process_image("./testing_imgs/a.jpg")
print(json_output)