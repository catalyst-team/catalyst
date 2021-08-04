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

    # image_id -> {
    #   file_name,
    #   height,
    #   width,
    #   annotations([{id, iscrowd, category_id, bbox}, ...])
    # }
    images = {}
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


def clip(values, min_value=0.0, max_value=1.0):
    return [min(max(num, min_value), max_value) for num in values]


class SSDDataset(Dataset):
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
            xyxy = clip(xyxy, 0.0, 1.0)
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


def draw_msra_gaussian(heatmap, channel, center, sigma=2):
    """Draw a gaussian on heatmap channel (inplace function).

    Args:
        heatmap (np.ndarray): heatmap matrix, expected shapes [C, W, H].
        channel (int): channel to use for drawing a gaussian.
        center (Tuple[int, int]): gaussian center coordinates.
        sigma (float): gaussian size. Default is ``2``.
    """
    tmp_size = sigma * 6
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    _, w, h = heatmap.shape
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = (max(0, -ul[0]), min(br[0], h) - ul[0])
    g_y = (max(0, -ul[1]), min(br[1], w) - ul[1])
    img_x = (max(0, ul[0]), min(br[0], h))
    img_y = (max(0, ul[1]), min(br[1], w))
    # fmt: off
    heatmap[channel, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[channel, img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
    )
    # fmt: on


class CenterNetDataset(Dataset):
    def __init__(self, coco_json_path, images_dir=None, transforms=None, down_ratio=4):
        self.file = coco_json_path
        self.img_dir = images_dir
        self.transforms = transforms
        self.down_ratio = down_ratio

        self.images, self.categories = load_coco_json(coco_json_path)
        self.images_list = sorted(self.images.keys())

        self.class_to_cid = {
            cls_idx: cat_id for cls_idx, cat_id in enumerate(sorted(self.categories.keys()))
        }
        self.cid_to_class = {v: k for k, v in self.class_to_cid.items()}
        self.num_classes = len(self.class_to_cid)
        self.class_labels = [
            self.categories[self.class_to_cid[cls_idx]]
            for cls_idx in range(len(self.class_to_cid))
        ]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_id = self.images_list[index]
        img_record = self.images[img_id]

        path = img_record["file_name"]
        if self.img_dir is not None:
            path = os.path.join(self.img_dir, path)
        image = read_image(path)
        original_size = [image.shape[0], image.shape[1]]  # height, width

        boxes = []  # each element is a tuple of (x1, y1, x2, y2, "class")
        for annotation in img_record["annotations"]:
            pixel_xywh = annotation["bbox"]
            # skip bounding boxes with 0 height or 0 width
            if pixel_xywh[2] == 0 or pixel_xywh[3] == 0:
                continue
            xyxy = pixels_to_absolute(
                pixel_xywh, width=img_record["width"], height=img_record["height"]
            )
            xyxy = clip(xyxy, 0.0, 1.0)
            bbox_class = str(self.cid_to_class[annotation["category_id"]])
            boxes.append(xyxy + [str(bbox_class)])

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            image, boxes = transformed["image"], transformed["bboxes"]
        else:
            image = torch.from_numpy((image / 255.0).astype(np.float32)).permute(2, 0, 1)

        labels = np.array([int(items[4]) for items in boxes])
        boxes = np.array([items[:4] for items in boxes], dtype=np.float32)
        # boxes = change_box_order(boxes, "xyxy2xywh")  # (x1, y1, x2, y2) -> (cx, cy, w, h)

        heatmap_height = image.shape[1] // self.down_ratio
        heatmap_width = image.shape[2] // self.down_ratio
        # draw class centers
        heatmap = np.zeros((self.num_classes, heatmap_height, heatmap_width), dtype=np.float32)
        for (x1, y1, x2, y2), cls_channel in zip(boxes, labels):
            w, h = abs(x2 - x1), abs(y2 - y1)
            xc, yc = x1 + w // 2, y1 + h // 2
            scaled_xc = int(xc * heatmap_width)
            scaled_yc = int(yc * heatmap_height)
            draw_msra_gaussian(
                heatmap, cls_channel, (scaled_xc, scaled_yc), sigma=np.clip(w * h, 2, 4)
            )
        # draw regression squares
        wh_regr = np.zeros((2, heatmap_height, heatmap_width), dtype=np.float32)
        regrs = boxes[:, 2:] - boxes[:, :2]  # width, height
        for r, (x1, y1, x2, y2) in zip(regrs, boxes):
            w, h = abs(x2 - x1), abs(y2 - y1)
            xc, yc = x1 + w // 2, y1 + h // 2
            scaled_xc = int(xc * heatmap_width)
            scaled_yc = int(yc * heatmap_height)
            for i in range(-2, 2 + 1):
                for j in range(-2, 2 + 1):
                    try:
                        a = max(scaled_xc + i, 0)
                        b = min(scaled_yc + j, heatmap_height)
                        wh_regr[:, a, b] = r
                    except:  # noqa: E722
                        pass
        wh_regr[0] = wh_regr[0].T
        wh_regr[1] = wh_regr[1].T

        return {
            "image": image,
            "original_size": original_size,
            "size": [image.size(1), image.size(2)],
            "heatmap": torch.from_numpy(heatmap),
            "wh_regr": torch.from_numpy(wh_regr),
            "bboxes": boxes,
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch):
        keys = list(batch[0].keys())
        packed_batch = {k: [] for k in keys}
        for element in batch:
            for k in keys:
                packed_batch[k].append(element[k])
        for k in ("image", "heatmap", "wh_regr"):
            packed_batch[k] = torch.stack(packed_batch[k], 0)
        return packed_batch
