# flake8: noqa

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_coco_json(path):
    """Read json with annotations.

    Args:
        path (str): path to .json file

    Raises:
        RuntimeError if .json file has no images
        RuntimeError if .json file has no categories

    Returns:
        images mapping and categories mapping
    """

    with open(path, "r") as in_file:
        content = json.load(in_file)

    if not len(content["images"]):
        raise RuntimeError(f"There is no image records in '{path}' file!")

    if not len(content["categories"]):
        raise RuntimeError(f"There is no categories in '{path}' file!")

    images = (
        {}
    )  # image_id -> {file_name, height, width, annotations([{id, iscrowd, category_id, bbox}, ...])}
    for record in content["images"]:
        images[record["id"]] = {
            "file_name": record["file_name"],
            "height": record["height"],
            "width": record["width"],
            "annotations": [],
        }

    categories = {}  # category_id -> name
    for record in content["categories"]:
        categories[record["id"]] = record["name"]

    for record in content["annotations"]:
        images[record["image_id"]]["annotations"].append(
            {
                "id": record["id"],
                "iscrowd": record["iscrowd"],
                "category_id": record["category_id"],
                "bbox": record["bbox"],
            }
        )

    return images, categories


def read_image(path):
    """Read image from given path.

    Args:
        path (str or Path): path to an image.

    Raises:
        FileNotFoundError when missing image file

    Returns:
        np.ndarray with image.
    """
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"There is no '{path}'!")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def pixels_to_absolute(box, width, height):
    """Convert pixel coordinates to absolute scales ([0,1]).

    Args:
        box (Tuple[number, number, number, number]): bounding box coordinates,
            expected list/tuple with 4 int values (x, y, w, h).
        width (int): image width
        height (int): image height

    Returns:
        List[float, float, float, float] with absolute coordinates (x1, y1, x2, y2).
    """
    x, y, w, h = box
    return [x / width, y / height, (x + w) / width, (y + h) / height]


class DetectionDataset(Dataset):
    def __init__(
        self,
        coco_json_path,
        images_dir=None,
        transforms=None,
        background_id=None,
        num_anchors=8732,
    ):
        self.file = coco_json_path
        self.images_dir = images_dir
        self.transforms = transforms
        self.num_anchors = num_anchors

        self.images, self.category_id2category_name = load_coco_json(self.file)
        self.images_list = sorted(self.images.keys())

        if background_id is None:
            self.background_class = 0
            self.class2category_id = {self.background_class: -1}
            for class_index, category_id in enumerate(
                sorted(self.category_id2category_name.keys()), start=1
            ):
                self.class2category_id[class_index] = category_id
            self.category_id2class = {v: k for k, v in self.class2category_id.items()}
        else:
            self.class2category_id = {
                cls_idx: cat_id
                for cls_idx, cat_id in enumerate(sorted(self.category_id2category_name.keys()))
            }
            self.category_id2class = {v: k for k, v in self.class2category_id.items()}
            self.background_class = self.category_id2class[background_id]

    @property
    def num_classes(self):
        return len(self.class2category_id)

    @property
    def class_labels(self):
        labels = []
        for cls_idx in range(len(self.class2category_id)):
            if cls_idx == self.background_class:
                labels.append("<BACKGROUND>")
            else:
                labels.append(self.category_id2category_name[self.class2category_id[cls_idx]])
        return labels

    @property
    def class_ids(self):
        # fmt: off
        return [
            self.class2category_id[class_number]
            for class_number in range(len(self.class2category_id))
        ]
        # fmt: on

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_id = self.images_list[index]
        img_record = self.images[img_id]

        path = img_record["file_name"]
        if self.images_dir is not None:
            path = os.path.join(self.images_dir, path)
        image = read_image(path)

        boxes = []  # each element is a tuple of (x1, y1, x2, y2, "class")
        for annotation in img_record["annotations"]:
            xyxy = pixels_to_absolute(
                annotation["bbox"], img_record["width"], img_record["height"]
            )
            assert all(
                0 <= num <= 1 for num in xyxy
            ), f"All numbers should be in range [0, 1], but got {xyxy}!"
            bbox_class = str(self.category_id2class[annotation["category_id"]])
            boxes.append(xyxy + [str(bbox_class)])

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            image, boxes = transformed["image"], transformed["bboxes"]
        else:
            image = torch.from_numpy((image / 255.0).astype(np.float32)).permute(2, 0, 1)

        bboxes = torch.zeros(size=(self.num_anchors, 4), dtype=torch.float)
        labels = torch.zeros(size=(self.num_anchors,), dtype=torch.long)
        for i, (x1, y1, x2, y2, label) in enumerate(boxes):
            w, h = x2 - x1, y2 - y1
            # (x_center, y_center, w, h)
            bboxes[i, 0] = x1 + w / 2
            bboxes[i, 1] = y1 + h / 2
            bboxes[i, 2] = w
            bboxes[i, 3] = h
            labels[i] = int(label)

        return {"image": image, "bboxes": bboxes, "labels": labels}
