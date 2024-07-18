# --------------------------------------------------------
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
#
# labelme2COCO.py
#
# reference from https://github.com/labelmeai/labelme/blob/main/examples/instance_segmentation/labelme2coco.py
# ----------------------------------------------------
import argparse
import os
import os.path as osp
import shutil
from tqdm import tqdm


from COCOformat import ImageInfo, Annotation, KeypointDB, COCOEncoder

import json
import glob
import labelme
import imgviz


def arg_parser():
    parser = argparse.ArgumentParser(description="labelme2COCO")
    parser.add_argument("--input_dir",
                        default='./data/coco/foot/',
                        help="input annotated your directory")
    parser.add_argument("--output_dir",
                        default='./data/coco/test/',
                        help="output dataset directory"
                        )
    parser.add_argument("--labels",
                        help="labels file",
                        required=False,
                        )

    parser.add_argument("--imgaug",
                        help="If you want to augment, This flag is True",
                        required=True,
                        )
    args = parser.parse_args()

    return args


class Labelme2COCOKeypointDB(KeypointDB):
    def __init__(self, args, json_file=None, is_load_coco=False):
        super().__init__(args, json_file=None, is_load_coco=False)
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir
        self.json_file_list = json_file

        if not is_load_coco:
            if self.json_file_list is None:
                raise Exception("Please Input Json file list")

            else:
                self._init_categories()
                self.generate_db()
        else:
            # _load_coco_json
            # Only One json_file
            self.load_coco_json()
            pass

    def _init_categories(self):
        """ Read First annotation Json file and apply base information for categories """

        categories = dict(
            keypoints=[],
            # skeletons=[[1, 2], [2, 3], [3, 16], [4, 5], [5, 6], [6, 16], [7, 8], [8, 9],
            #            [9, 16], [10, 11], [11, 12], [12, 16], [13, 14], [14, 15], [15, 16], [16, 17]],
            skeletons=[[1, 2], [2, 3], [3, 15], [4, 5], [5, 6], [6, 15], [7, 8], [8, 9], [9, 15],
                       [10, 11], [11, 9], [12, 13], [13, 14], [14, 17], [15, 16]],
            id=1,
            name="foot",
            supercategory="foot"
        )

        shapes = labelme.LabelFile(filename=self.json_file_list[0]).shapes
        for s in shapes:
            if s['label'] == "Foot" or s['label'] == "foot":
                continue
            categories['keypoints'].append(s["label"])

        self.db['categories'].append(categories)
        print("Initialize Basic categories")
        return

    def generate_db(self):
        print("Generating Keypoint DB")
        print("=============================================")

        for image_id, filename in tqdm(enumerate(self.json_file_list)):
            print("Generating Dataset from:", filename)

            # label_file = labelme.LabelFile(filename=filename)
            with open(filename, 'r') as file:
                data = json.load(file)

                origin_image_path = osp.join(self.input_dir, data.get("imagePath"))
                out_img_file = osp.join(self.output_dir, "Images", str(image_id) + ".png")

                shutil.copy(src=origin_image_path, dst=out_img_file)

                image_info = ImageInfo(
                    license=0,
                    coco_url='',
                    flickr_url='',
                    file_name=osp.relpath(out_img_file, osp.dirname(out_img_file)),
                    height=data.get("imageHeight"),
                    width=data.get("imageWidth"),
                    date_captured=None,
                    id=image_id,
                )
                shapes = data['shapes']
                annotation_db = Annotation(
                    area=0,
                    iscrowd=0,
                    image_id=image_id,
                    bbox=[],
                    category_id=1,
                    id=image_id,
                    keypoints=[],
                    num_keypoints=17,
                )
                num_keypoints = 0
                for s in shapes:
                    if s["shape_type"] == "rectangle":
                        x0, y0 = s.get("points")[0]
                        x1, y1 = s.get("points")[1]
                        annotation_db.bbox = [x0, y0, x1 - x0, y1 - y0]
                        annotation_db.area = annotation_db.bbox[2] * annotation_db.bbox[3]
                        continue
                    num_keypoints += 1
                    point_coord = [s.get("points")[0][0], s.get("points")[0][1], 2]
                    annotation_db.keypoints.extend(point_coord)
                annotation_db.num_keypoints = num_keypoints
                self.db['images'].append(image_info)
                self.db['annotations'].append(annotation_db)
            #
            # except FileNotFoundError:
            #     print(f"Files {filename} not found")

    def augmentation(self):
        print("Augmentation DB")
        print("=============================================")


def main():
    args = arg_parser()

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not osp.exists(osp.join(args.output_dir, "Images")):
        os.makedirs(osp.join(args.output_dir, "Images"))

    print("Creating Dataset to: ", args.output_dir)
    print("=============================================")

    db = Labelme2COCOKeypointDB(args, json_file=glob.glob(osp.join(args.input_dir, "*.json")))
    db.saver()


if __name__ == '__main__':
    main()
