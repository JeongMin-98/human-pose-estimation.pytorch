# --------------------------------------------------------
#
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
#
# ----------------------------------------------------

import argparse
import json
import os
import re

from pycocotools.coco import COCO


def arg_parser():
    parser = argparse.ArgumentParser(description='Filter File')
    parser.add_argument('--json',
                        default='data/coco/annotations/RHPE_anatomical_ROIs_train.json',
                        help='path/to/annotations.json',
                        required=False,
                        type=str)

    parser.add_argument('--output',
                        default='output',
                        help='path/to/output_json.json',
                        type=str)

    args = parser.parse_args()
    return args


def filter_images(args):
    json_path = args.json
    output_path = args.output

    coco = COCO(json_path)
    exclude_images = {'00138.png', '00581.png', '00670.png', '01024.png'}
    pattern = re.compile(r'(0000[1-9]|000[1-9][0-9]|00[1-9][0-9]{2}|0100[0-9]|0101[0-9]|0102[0-4]).png')

    all_images = coco.loadImgs(coco.getImgIds())

    filtered_images = [
        img for img in all_images
        if pattern.match(img['file_name']) and img['file_name'] not in exclude_images
    ]

    filtered_image_ids = {img['id'] for img in filtered_images}

    filtered_annotations = coco.loadAnns(coco.getAnnIds(imgIds=list(filtered_image_ids)))

    categories = coco.loadCats(coco.getCatIds())

    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': categories
    }

    with open(output_path, 'w') as file:
        json.dump(filtered_data, file, indent=4)

    print(f"Filtered annotations have been saved to '{output_path}'")


if __name__ == '__main__':
    args = arg_parser()
    filter_images(args)
