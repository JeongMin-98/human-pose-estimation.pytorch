# --------------------------------------------------------
#
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
#
# ----------------------------------------------------

import argparse
import json
import random
import os
import re

from pycocotools.coco import COCO

random.seed(42)


def arg_parser():
    parser = argparse.ArgumentParser(description='Filter File')
    parser.add_argument('--json',
                        default='data/coco/annotations/RHPE_anatomical_ROIs_train.json',
                        help='path/to/annotations.json',
                        required=False,
                        type=str)

    parser.add_argument('--prefix',
                        default='output',
                        help='prefix of output files',
                        type=str)

    parser.add_argument('--split',
                        default=True,
                        help='split yes or no True, False',
                        type=bool)

    args = parser.parse_args()
    return args


def make_filtered_data_db(images, annotations, categories):
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }


def split_data(filtered_images, filtered_annotations, category):
    random.shuffle(filtered_images)
    train_size = int(0.8 * len(filtered_images))
    print(f"split filtered_images and filtered_annotations || ALL: {len(filtered_images)} || train : {train_size}")

    train_images = filtered_images[:train_size]
    val_images = filtered_images[train_size:]

    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}

    train_annotations = [ann for ann in filtered_annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in filtered_annotations if ann['image_id'] in val_image_ids]

    train_data = make_filtered_data_db(train_images, train_annotations, category)
    val_data = make_filtered_data_db(val_images, val_annotations, category)

    return train_data, val_data


def saver(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"save json -> {path}")


def filter_images(args):
    json_path = args.json
    output_path = args.prefix
    train = output_path + '_train.json'
    test = output_path + '_val.json'

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

    filtered_data = make_filtered_data_db(filtered_images, filtered_annotations, categories)

    with open(output_path+'.json', 'w') as file:
        json.dump(filtered_data, file, indent=4)

    print(f"Filtered annotations have been saved to '{output_path}'")

    train_data, val_data = split_data(filtered_images, filtered_annotations, categories)

    saver(train, train_data)
    saver(test, val_data)


if __name__ == '__main__':
    args = arg_parser()
    filter_images(args)
