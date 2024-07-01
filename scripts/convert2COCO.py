# --------------------------------------------------------
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
#
# I will implement a code that coverts my custom annotation json to COCO FORMAT. (WIP)
#
# ----------------------------------------------------

import json


def json_to_coco_format(json_files):
    images = []
    annotations = []
    categories = []  # Assuming you have specific categories, you can define them here

    image_id = 0
    annotation_id = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_info = {
            'id': image_id,
            'file_name': data['imagePath'],  # Assuming imagePath contains the filename
            'width': data['imageWidth'],
            'height': data['imageHeight']
        }
        images.append(image_info)

        for shape in data['shapes']:
            category_id = categories.index(shape['label']) if shape['label'] in categories else len(categories)
            categories.append(shape['label'])

            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [shape['points'][0][0], shape['points'][0][1], 0, 0],  # Modify as per your data
                'area': 0,  # Calculate area if available
                'iscrowd': 0,  # Assuming not crowded
                'segmentation': [],  # Depending on your data
                'attributes': {}  # Any additional attributes
            }
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1

    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i, 'name': cat} for i, cat in enumerate(categories)]
    }

    with open('coco_format.json', 'w') as f:
        json.dump(coco_data, f, indent=4)

    print('COCO format file saved as coco_format.json')


# Replace with your actual JSON file paths
json_files = [
    'path/to/your/json_file1.json',
    'path/to/your/json_file2.json',
    # Add more JSON files as needed
]

json_to_coco_format(json_files)
