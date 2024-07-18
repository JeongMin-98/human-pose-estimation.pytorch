import argparse
import json

import os
import numpy as np
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage, Keypoint, KeypointsOnImage
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
# from convert2COCO import Labelme2COCOKeypointDB
from COCOformat import ImageInfo, Annotation, KeypointDB

# 증강할 이미지 개수
num_augmented_images = 5
# 증강 시퀀스 정의
seq = iaa.Sequential([
    # iaa.Fliplr(0.5),  # 좌우 반전
    # iaa.Affine(scale=(0.5, 1.5)),  # 스케일링
    iaa.Rotate((-90, 90)),  # 밝기 조절
    # iaa.GaussianBlur(sigma=(0.0, 3.0))  # 가우시안 블러
])


def arg_parser():
    parser = argparse.ArgumentParser(description="Image augmentation")
    parser.add_argument("--input_dir",
                        default="../data/coco/imgaug/cocot/Images",
                        help="input annotated your directory")
    parser.add_argument("--output_dir",
                        default="../data/coco/imgaugtest",
                        help="output dataset directory"
                        )
    parser.add_argument("--json",
                        default='../data/coco/imgaug/cocot/annotations.json')
    args = parser.parse_args()
    return args


# def append_annotations(db: Labelme2COCOKeypointDB, keypoints: list[Keypoint], bboxes: list[BoundingBox]):
#     temp_keypoint = []
#     temp_bbox = []
#
#     for keypoint in keypoints:
#         x, y, v = keypoint.x, keypoint.y, 2
#         temp_keypoint.extend([x, y, v])
#     for bbox in bboxes:
#         temp_bbox.extend([bbox.x1, bbox.y1, bbox.width, bbox.height])
#
#     # db must be loaded.
#     if db.load_coco is False:
#         raise Exception("DB must be loaded!!")

def convert_to_color(image):
    if image.ndim == 2:  # Check if the image is grayscale
        image = np.stack((image,) * 3, axis=-1)  # Convert to 3D by stacking the grayscale values
    return image


def load_json_before_imgaug(args):
    input_image_dir = args.input_dir
    input_keypoints_file = os.path.join(args.json)
    # input_keypoints_file = "/home/mlpa_jm/PoseXray/data/Foot_Data/test_out/annotations.json"
    # input_keypoints_file = "../data/coco/Foot_ann(D)_new/Foot_Doctor_n1_annotations.json"
    output_image_dir = args.output_dir
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    else:
        pass

    aug_db = KeypointDB(args, json_file=input_keypoints_file, is_load_coco=True)

    with open(input_keypoints_file, "r") as f:
        keypoints_data = json.load(f)
    aug_db.db['categories'] = keypoints_data['categories']
    last_image_id = keypoints_data['images'][-1]["id"]
    last_annotation_id = keypoints_data['annotations'][-1]["id"]
    pbar = tqdm(len(keypoints_data['images']))
    for i, image_info in enumerate(keypoints_data['images']):
        image_path = os.path.join(input_image_dir, image_info['file_name'])
        image = convert_to_color(imageio.v3.imread(image_path))

        original_image_info = deepcopy(keypoints_data['images'][i])
        original_image_info["id"] = last_image_id
        original_image_info["file_name"] = image_info['file_name']
        aug_db.db['images'].append(original_image_info)
        original_annotation_info = deepcopy(keypoints_data['annotations'][i])
        original_annotation_info["image_id"] = last_image_id
        original_annotation_info["id"] = last_annotation_id
        aug_db.db['annotations'].append(original_annotation_info)

        last_image_id += 1
        last_annotation_id += 1

        bbox = keypoints_data['annotations'][i]['bbox']
        bbox = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3], label='foot')

        bbox_on_image = BoundingBoxesOnImage([bbox], shape=image.shape)

        keypoints = keypoints_data['annotations'][i]['keypoints']
        keypoint_bucket = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if v > 0:
                keypoint_bucket.append(Keypoint(x=x, y=y))

        keypoint_on_image = KeypointsOnImage(keypoint_bucket, shape=image.shape)
        pbar.update(1)

        for j in range(5):
            image_aug, bbox_aug, keypoints_aug = seq(image=image, bounding_boxes=bbox_on_image,
                                                     keypoints=keypoint_on_image)

            # file save
            new_file_name = f"{last_image_id - 1}{j}.png"
            new_image_path = os.path.join(output_image_dir, new_file_name)
            imageio.v3.imwrite(new_image_path, image_aug)

            # image_after_aug = bbox_aug.draw_on_image(image_aug, size=2, color=[0, 0, 225])
            # plt.imshow(image_after_aug)
            # plt.show()
            aug_db.db['images'].append(ImageInfo(
                license=0,
                coco_url='',
                flickr_url='',
                file_name=os.path.relpath(new_file_name, os.path.dirname(new_file_name)),
                height=image_aug.shape[0],
                width=image_aug.shape[1],
                date_captured=None,
                id=last_image_id,
            ))

            temp_bbox_aug = bbox_aug.bounding_boxes[0]

            # There is only one bbox.
            aug_db.db['annotations'].append(Annotation(
                area=temp_bbox_aug.area,
                iscrowd=0,
                image_id=last_image_id,
                bbox=[],
                keypoints=[],
                num_keypoints=17,
                id=last_annotation_id,
                category_id=1,
            ))

            aug_db.db['annotations'][-1].bbox.extend(
                [temp_bbox_aug.x1, temp_bbox_aug.y1, temp_bbox_aug.width, temp_bbox_aug.height])

            for keypoint in keypoints_aug.keypoints:
                aug_db.db['annotations'][-1].keypoints.extend([keypoint.x, keypoint.y, 2])

            last_image_id += 1
            last_annotation_id += 1
    pbar.close()
    f.close()
    aug_db.saver()


if __name__ == "__main__":
    args = arg_parser()
    load_json_before_imgaug(args)
