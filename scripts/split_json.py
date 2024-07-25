import random

from pycocotools.coco import COCO
from filtered_image import arg_parser, split_data, saver

random.seed(42)


def split(args):
    json_path = args.json
    output_path = args.prefix
    train = output_path + '_train.json'
    val = output_path + '_val.json'
    test = output_path + '_test.json'

    coco = COCO(json_path)

    all_images = coco.loadImgs(coco.getImgIds())
    all_annotations = coco.loadAnns(coco.getAnnIds())
    categories = coco.loadCats(coco.getCatIds())

    train_data, val_data, test_data = None, None, None
    if args.split:
        train_data, val_data, test_data = split_data(all_images, all_annotations, categories)

    saver(train, train_data)
    saver(val, val_data)
    saver(test, test_data)


if __name__ == '__main__':
    args = arg_parser()
    split(args)
