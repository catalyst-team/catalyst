import json
import os
import sys

import xmltodict


def load_annotations(file):
    """Read annotations from file.

    Args:
        file (str): path to file.

    Returns:
        List[Dict[str, Any]] with bounding boxes
    """
    with open(file, "r") as in_file:
        content = xmltodict.parse(in_file.read())
    filename = content["annotation"]["filename"]
    width = int(content["annotation"]["size"]["width"])
    height = int(content["annotation"]["size"]["height"])
    objects = content["annotation"]["object"]
    objects = [objects] if isinstance(objects, dict) else objects
    annots = []
    for item in objects:
        annots.append(
            {
                "category": item["name"].lower().strip(),
                "x1": int(item["bndbox"]["xmin"]),
                "y1": int(item["bndbox"]["xmin"]),
                "x2": int(item["bndbox"]["xmax"]),
                "y2": int(item["bndbox"]["ymax"]),
            }
        )
    return filename, (width, height), annots


def main():
    """Convert data to COCO format."""
    imgs_dir, annots_dir, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    category2id = {"banana": 1, "snake fruit": 2, "dragon fruit": 3, "pineapple": 4}
    categories = [{"id": cat_id, "name": cat_name} for cat_name, cat_id in category2id.items()]
    images = []
    annotations = []
    img_id = 1
    annot_id = 1
    for img_file in os.listdir(imgs_dir):
        if not img_file.endswith(".png"):
            continue
        annot_file = os.path.join(annots_dir, img_file[:-4] + ".xml")
        filename, (width, height), annots = load_annotations(annot_file)
        images.append({"id": img_id, "file_name": filename, "width": width, "height": height})
        for item in annots:
            cat_id = category2id[item["category"]]
            x1, y1 = min(item["x1"], item["x2"]), min(item["y1"], item["y2"])
            x2, y2 = max(item["x1"], item["x2"]), max(item["y1"], item["y2"])
            area = (x2 - x1) * (y2 - y1)
            annotations.append(
                {
                    "id": annot_id,
                    "image_id": img_id,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": cat_id,
                }
            )
            annot_id += 1
        img_id += 1

    with open(output_file, "w") as out_file:
        json.dump(
            {"categories": categories, "images": images, "annotations": annotations},
            out_file,
            indent=2,
        )


if __name__ == "__main__":
    main()
