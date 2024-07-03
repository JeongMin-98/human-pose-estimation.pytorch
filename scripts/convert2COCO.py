# --------------------------------------------------------
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
#
# I will implement a code that coverts my custom annotation json to COCO FORMAT. (WIP)
#
# ----------------------------------------------------
import argparse
import os
import os.path as osp
import shutil

from collections import defaultdict

import json
import glob
import labelme
import imgviz


def arg_parser():
    parser = argparse.ArgumentParser(description="labelme2COCO")
    parser.add_argument("input_dir", help="input annotated your directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    return args


class KeypointDB:
    def __init__(self, args, json_file_list=None):
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir
        self.json_file_list = json_file_list
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

        if self.json_file_list is None:
            raise Exception("Please Input Json file list")

        else:
            self._init_categories()
            self.generate_db()

    def _init_categories(self):
        """ Read First annotation Json file and apply base information for categories """

        categories = dict(
            keypoints=[],
            skeletons=[[1, 2], [2, 3], [3, 16], [4, 5], [5, 6], [6, 16], [7, 8], [8, 9],
                       [9, 16], [10, 11], [11, 12], [12, 16], [13, 14], [14, 15], [15, 16], [16, 17]],
            id=1,
            name="foot",
            supercategory="hand"
        )

        shapes = labelme.LabelFile(filename=self.json_file_list[0]).shapes
        for s in shapes:
            categories['keypoints'].append(s["label"])

        self.db['categories'].append(categories)
        print("Initialize Basic categories")
        return

    def generate_db(self):
        print("Generating Keypoint DB")
        print("=============================================")
        for image_id, filename in enumerate(self.json_file_list):
            print("Generating Dataset from:", filename)

            # label_file = labelme.LabelFile(filename=filename)
            try:
                with open(filename, 'r') as file:
                    data = json.load(file)

                    out_img_file = osp.join(self.output_dir, "Image", str(image_id) + ".png")

                    shutil.copy(data.get("imagePath"), out_img_file)

                    dict(
                        license=0,
                        url=None,
                        file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                        height=data.get("imageHeight"),
                        width=data.get("imageWidth"),
                        date_captured=None,
                        id=image_id,
                    )

            except FileNotFoundError:
                print(f"Files {filename} not found")





def main():
    args = arg_parser()

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "Images"))

    print("Creating Dataset to: ", args.output_dir)
