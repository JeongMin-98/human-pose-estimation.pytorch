# --------------------------------------------------------
#
# Written by Jeongmin Kim(jm.kim@dankook.ac.kr)
#
# ----------------------------------------------------

import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description='visualize ground truth image (w/o flip,rotate)')
    parser.add_argument('--json',
                        default='data/coco/annotations/RHPE_anatomical_ROIs_train.json',
                        help='path/to/annotations.json',
                        required=False,
                        type=str)
    parser.add_argument('--data',
                        default='data/coco/images/RHPE_TRAIN',
                        help='path/to/images',
                        required=False,
                        type=str)

    parser.add_argument('--output',
                        default='output',
                        help='path/to/images',
                        type=str)

    args = parser.parse_args()

    return args


def vis_gt_image(args):
    # COCO annotation 파일 경로와 이미지 경로
    annotation_file = args.json
    image_folder = args.data
    output_dir = args.output
    output_dir = os.path.join(output_dir, 'scripts', image_folder.split('/')[-1])

    os.makedirs(output_dir, exist_ok=True)
    # COCO 객체 생성
    coco = COCO(annotation_file)

    # 이미지 ID 가져오기
    image_ids = coco.getImgIds()

    # 배치 사이즈 설정
    batch_size = 64
    rows = 8
    cols = 8


    # 그리드 설정
    for i in range(0, len(image_ids), batch_size):
        batch_image_ids = image_ids[i:i + batch_size]
        fig, axes = plt.subplots(rows, cols, figsize=(20, 30))
        axes = axes.flatten()

        for j, image_id in tqdm(enumerate(batch_image_ids)):
            # load image info
            image_info = coco.loadImgs(image_id)[0]
            image_path = os.path.join(image_folder, image_info['file_name'])

            # load image
            image = Image.open(image_path).convert('RGB')

            # ax setting
            ax = axes[j]
            ax.imshow(image)
            ax.axis('off')

            # load annotations for image
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(annotation_ids)
            for annotation in annotations:
                if 'keypoints' in annotation:
                    keypoints = annotation['keypoints']
                    for k in range(0, len(keypoints), 3):
                        keypoint_x = keypoints[k]
                        keypoint_y = keypoints[k + 1]
                        keypoint_v = keypoints[k + 2]
                        if keypoint_v > 0:  # v == 0: not labeled; v == 1: labeled but not visible; v == 2: labeled and visible
                            ax.plot(keypoint_x, keypoint_y, 'go' if keypoint_v == 2 else 'ro')
                # # 카테고리 이름 표시
                # category_id = annotation['category_id']
                # category_name = coco.loadCats(category_id)[0]['name']
                # ax.text(x, y - 10, category_name, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            ax.set_title(image_info['file_name'], fontsize=10)

        # plt.tight_layout()
        plt.savefig(fname=f'{output_dir}/{i}.png', format='png')
        print(f"save plot -> {output_dir}/{i}.png")
        # plt.tight_layout()
        # plt.show()


def main():
    args = arg_parser()
    vis_gt_image(args)
    return


if __name__ == '__main__':
    main()
