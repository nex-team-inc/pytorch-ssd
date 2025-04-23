import numpy as np
import logging
import pathlib
import cv2
import os
from pycocotools.coco import COCO


class COCOWholeBodyDataset:
    """Dataset for COCO Wholebody data with hand keypoints and bounding boxes."""

    def __init__(self, root, transform=None, target_transform=None, usage='train', 
                 keep_difficult=False):
        """
        Args:
            root: the root of the COCO dataset containing images directory
            transform: image transform
            target_transform: bbox transform
            keep_difficult: whether to keep difficult samples
            usage: 'train' or 'val'
        """
        self.root = pathlib.Path(root) / 'COCO'
        self.img_dir = self.root / 'images'
        self.ann_dir = self.root / 'annotations' / f'coco_wholebody_{usage}_v1.0.json'
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult

        self.coco = COCO(self.ann_dir)
        self.person_cat_id = self.coco.getCatIds(catNms=['person'])[0]
        
        # Filter to get only images with valid hand keypoints
        all_img_ids = self.coco.getImgIds(catIds=[self.person_cat_id])
        self.ids = self._filter_hand_images(all_img_ids)
        
        print(f"COCO Wholebody Dataset initialized with {len(self.ids)} images containing valid hand annotations.")
        
        # Set class names - we only have one class for hand detection
        self.class_names = ('BACKGROUND', 'hand')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def _filter_hand_images(self, img_ids):
        """Filter images to only include those with valid hand annotations."""
        valid_img_ids = []

        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id])
            if not ann_ids:
                continue

            has_valid_hands = False
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                left_valid = ann.get('lefthand_valid', False)
                right_valid = ann.get('righthand_valid', False)
                if left_valid or right_valid:
                    valid_img_ids.append(img_id)
                    break

        return valid_img_ids

    def __getitem__(self, index):
        img_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(img_id)

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]

        image = self._read_image(img_id)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

    def get_image(self, index):
        img_id = self.ids[index]
        image = self._read_image(img_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        img_id = self.ids[index]
        return img_id, self._get_annotation(img_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, img_id):
        """Get hand bounding boxes from annotations."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        is_difficult = []

        for ann in anns:
            # Check for valid left hand
            if ann.get('lefthand_valid', False) and 'lefthand_box' in ann:
                x, y, width, height = ann['lefthand_box']
                boxes.append([int(x), int(y), int(x + width), int(y + height)])
                labels.append(self.class_dict['hand'])
                is_difficult.append(0)
                
            # Check for valid right hand
            if ann.get('righthand_valid', False) and 'righthand_box' in ann:
                x, y, width, height = ann['righthand_box']
                boxes.append([int(x), int(y), int(x + width), int(y + height)])
                labels.append(self.class_dict['hand'])
                is_difficult.append(0)

        if boxes:
            return (np.array(boxes, dtype=np.float32),
                    np.array(labels, dtype=np.int64),
                    np.array(is_difficult, dtype=np.uint8))
        else:
            return (np.array([], dtype=np.float32).reshape(0, 4),
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.uint8))

    def _read_image(self, img_id):
        """Read image from file."""
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = str(self.img_dir / img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image