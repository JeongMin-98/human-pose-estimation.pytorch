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

    args = parser.parse_args()

    return args


def vis_gt_image(args):
    # COCO annotation 파일 경로와 이미지 경로
    annotation_file = args.json
    image_folder = args.data
    # COCO 객체 생성
    coco = COCO(annotation_file)

    # 이미지 ID 가져오기
    image_ids = coco.getImgIds()

    # 배치 사이즈 설정
    batch_size = 32
    rows = 8
    cols = 4

    # 그리드 설정
    fig, axes = plt.subplots(rows, cols, figsize=(20, 40))
    axes = axes.flatten()

    for i, image_id in enumerate(image_ids[:batch_size]):
        # load image info
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(image_folder, image_info['file_name'])

        # load image
        image = Image.open(image_path)

        # ax setting
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')

        # load annotations for image
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        for annotation in annotations:
            bbox = annotation['bbox']
            # COCO 형식: [x, y, width, height]
            x, y, width, height = bbox
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # 카테고리 이름 표시
            category_id = annotation['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            ax.text(x, y - 10, category_name, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # 빈 축 제거
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def main():
    args = arg_parser()
    vis_gt_image(args)
    return


if __name__ == '__main__':
    main()
