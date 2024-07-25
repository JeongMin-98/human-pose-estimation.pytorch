# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------
import json
import numpy as np
from dataclasses import dataclass
from os import path as osp
from typing import List, Tuple


@dataclass
class ImageInfo():
    license: int
    coco_url: str
    flickr_url: str
    file_name: str
    height: Tuple[int, float]
    width: Tuple[int, float]
    date_captured: None
    id: int


@dataclass
class Annotation:
    area: float
    iscrowd: int
    image_id: int
    bbox: List[float]
    category_id: int
    id: int
    keypoints: List[float]
    num_keypoints: int


class COCOEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (ImageInfo, Annotation)):
            return o.__dict__
        if isinstance(o, np.float32):
            return float(o)
        return super().default(o)


class KeypointDB:
    def __init__(self, args, json_file=None, is_load_coco=False):
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir
        self.json_file_list = json_file
        self.db = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year="",
                contributor="",
                date_created="",
            ),
            licenses=[
                dict(
                    url=None,
                    id=0,
                    name=None,
                )
            ],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

    def load_coco_json(self):
        # If you load coco json file, json file is only one file
        with open(self.json_file_list, 'r') as file:
            data = json.load(file)
            self.db = data
        return

    def saver(self):
        with open(osp.join(self.output_dir, "annotations.json"), 'w') as f:
            json.dump(self.db, f, indent=4, cls=COCOEncoder)

        print(f"save json -> {osp.join(self.output_dir, 'annotations.json')}")
