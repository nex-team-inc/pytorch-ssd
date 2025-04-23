import numpy as np
import logging
import pathlib
import json
import cv2
import os
from glob import glob


class CMUPDManualDataset:
    """Dataset for CMUPD_manual data with hand keypoints."""

    def __init__(self, root, transform=None, target_transform=None, usage='train', 
                 keep_difficult=False, bbox_padding=5):
        """
        Args:
            root: the root of the CMUPD_manual dataset
            transform: image transform
            target_transform: bbox transform
            usage: 'train' or 'test'
            keep_difficult: whether to keep difficult samples
            bbox_padding: padding around hand keypoints when creating bbox
        """
        self.root = pathlib.Path(root) / "CMUPD_manual"
        self.transform = transform
        self.target_transform = target_transform
        self.bbox_padding = bbox_padding
        
        # Determine train or test folder
        folder = "manual_test" if usage == 'test' else "manual_train"
        self.data_path = self.root / folder
        
        # Get all image file paths
        image_paths = sorted(glob(str(self.data_path / "*.jpg")))
        image_groups = {}
        for path in image_paths:
            filename = os.path.basename(path)
            parts = filename.split('_')
            if len(parts[-2]) == 2 and parts[-2].isdigit(): # indicate index of person in image
                # base_name = '_'.join(parts[:-2])
                continue # skipping multi-person images
            else:
                base_name = '_'.join(parts[:-1]) # everything before handedness indicator
            
            if base_name not in image_groups:
                image_groups[base_name] = []
            image_groups[base_name].append(path)
        
        self.unique_images = sorted(image_groups.keys())
        self.image_annotations = image_groups

        self.keep_difficult = keep_difficult
        
        # Set class names - we only have one class for hand detection
        self.class_names = ('BACKGROUND', 'hand')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        print(f"CMUPD Dataset initialized with {len(self.unique_images)} images.")

    def __getitem__(self, index):
        base_name = self.unique_images[index]
        image_paths = self.image_annotations[base_name]
        print(f"image_paths: {image_paths}")

        image = self._read_image(image_paths[0])
        image_height, image_width = image.shape[:2]
        all_boxes = []
        all_labels = []
        all_difficulties = []

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            image_id = os.path.splitext(filename)[0]
            boxes, labels, is_difficult = self._get_annotation(image_id, image_height, image_width)

            if not self.keep_difficult:
                valid_idx = is_difficult == 0
                boxes = boxes[valid_idx]
                labels = labels[valid_idx]
                is_difficult = is_difficult[valid_idx]

            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_labels.append(labels)
                all_difficulties.append(is_difficult)

        if all_boxes:
            boxes = np.vstack(all_boxes)
            labels = np.concatenate(all_labels)
            is_difficult = np.concatenate(all_difficulties)
        else:
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            labels = np.array([], dtype=np.int64)
            is_difficult = np.array([], dtype=np.uint8)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

    def get_image(self, index):
        base_name = self.unique_images[index]
        image_paths = self.image_annotations[base_name]
        image = self._read_image(image_paths[0])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        base_name = self.unique_images[index]
        return base_name, self._get_all_annotations(base_name)

    def _get_all_annotations(self, base_name):
        image_paths = self.image_annotations[base_name]
        all_boxes = []
        all_labels = []
        all_difficulties = []
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            image_id = os.path.splitext(filename)[0]
            image = self._read_image(img_path)
            image_height, image_width = image.shape[:2]
            boxes, labels, is_difficult = self._get_annotation(image_id, image_height, image_width)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_labels.append(labels)
                all_difficulties.append(is_difficult)
        
        if all_boxes:
            return np.vstack(all_boxes), np.concatenate(all_labels), np.concatenate(all_difficulties)
        else:
            return (np.array([], dtype=np.float32).reshape(0, 4),
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.uint8))

    def __len__(self):
        return len(self.unique_images)

    def _get_annotation(self, image_id, image_height, image_width):
        """Get bounding boxes and labels from annotations."""
        annotation_file = self.data_path / f"{image_id}.json"
        print(f"annotation_file: {annotation_file}")
        
        # Load annotation
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)

        hand_pts = np.array(annotation["hand_pts"])
        
        # Check valid keypoints
        if len(hand_pts) == 0 or np.all(hand_pts[:, 2] == 0):
            return (np.array([], dtype=np.float32),
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.uint8))

        valid_pts = hand_pts[hand_pts[:, 2] > 0, :2]

        x_min, y_min = np.min(valid_pts, axis=0)
        x_max, y_max = np.max(valid_pts, axis=0)

        # add padding
        x_min = max(0, x_min - self.bbox_padding)
        y_min = max(0, y_min - self.bbox_padding)
        x_max = min(image_width - 1, x_max + self.bbox_padding)
        y_max = min(image_height - 1, y_max + self.bbox_padding)

        boxes = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        labels = np.array([self.class_dict['hand']], dtype=np.int64)
        is_difficult = np.array([0], dtype=np.uint8)
        
        return boxes, labels, is_difficult

    def _read_image(self, image_path):
        """Read image from file."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image