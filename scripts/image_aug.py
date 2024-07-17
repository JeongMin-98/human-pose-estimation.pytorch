import os
import json
import argparse
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage, Keypoint, KeypointsOnImage
import numpy as np
from convert2COCO import KeypointDB, ImageInfo, Annotation

# 증강할 이미지 개수
num_augmented_images = 5
# 증강 시퀀스 정의
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 좌우 반전
    # iaa.Affine(scale=(0.5, 1.5)),  # 스케일링
    iaa.Rotate((-45, 45)),  # 밝기 조절
    # iaa.GaussianBlur(sigma=(0.0, 3.0))  # 가우시안 블러
])


def arg_parser():
    parser = argparse.ArgumentParser(description="Image augmentation")
    parser.add_argument("--input_dir",
                        default="/home/jeongmin/human-pose-estimation.pytorch/data/coco/Foot_ann(D)_new/Images",
                        help="input annotated your directory")
    parser.add_argument("--output_dir",
                        default="/home/jeongmin/human-pose-estimation.pytorch/data/coco/imgaugtest",
                        help="output dataset directory"
                        )
    parser.add_argument("--json",
                        default='your.json')
    args = parser.parse_args()
    return args


def append_annotations(db: KeypointDB, keypoints: list[Keypoint], bboxes: list[BoundingBox]):
    temp_keypoint = []
    temp_bbox = []

    for keypoint in keypoints:
        x, y, v = keypoint.x, keypoint.y, 2
        temp_keypoint.extend([x, y, v])
    for bbox in bboxes:
        temp_bbox.extend([bbox.x1, bbox.y1, bbox.width, bbox.height])

    # db must be loaded.
    if db.load_coco is False:
        raise Exception("DB must be loaded!!")


def load_json_before_imgaug(args):
    input_image_dir = args.input_dir
    # input_keypoints_file = os.path.join(input_image_dir, args.json)
    input_keypoints_file = "/home/jeongmin/human-pose-estimation.pytorch/data/coco/Foot_ann(D)_new/Foot_Doctor_n1_annotations.json"
    output_image_dir = args.output_dir
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    aug_db = KeypointDB(args, json_file=input_keypoints_file, is_load_coco=True)

    with open(input_keypoints_file, "r") as f:
        keypoints_data = json.load(f)
    for i, image_info in enumerate(keypoints_data['images']):
        image_path = os.path.join(input_image_dir, image_info['file_name'])
        image = imageio.v3.imread(image_path)

        bbox = keypoints_data['annotations'][i]['bbox']
        bbox = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3], label='foot')
        print(bbox.coords)

        bbox_on_image = BoundingBoxesOnImage([bbox], shape=image.shape)

        keypoints = keypoints_data['annotations'][i]['keypoints']
        keypoint_bucket = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if v > 0:
                keypoint_bucket.append(Keypoint(x=x, y=y))

        keypoint_on_image = KeypointsOnImage(keypoint_bucket, shape=image.shape)

        for j in range(5):
            image_aug, bbox_aug, keypoints_aug = seq(image=image, bounding_boxes=bbox_on_image,
                                                     keypoints=keypoint_on_image)

            # prepare image_info

            # file save
            new_file_name = f"{image_info['file_name']}_aug_{j}.jpg"
            new_image_path = os.path.join(output_image_dir, new_file_name)
            imageio.v3.imwrite(new_image_path, image_aug)


if __name__ == "__main__":
    args = arg_parser()
    load_json_before_imgaug(args)
