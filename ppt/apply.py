import os
import cv2
import json
import numpy as np
from pprint import pprint

colors = ((0, 255, 0), (0, 0, 255))     # colors that we use to mark the detections
image_width, image_height = 768, 512    # the size of the image
part_size = image_height // 2           # the size of the zoomed thing

def get_key(filename):
    if '_' in filename:
        new_key = filename.split("_")[0]
    else:
        new_key = filename.split(".")[0]
    return new_key

'''
    new_data = dict()
    for key, item in data.items():
        new_key = get_key(key)
        new_data[new_key] = data[key]
'''

def process_dataset(name, marks_file, methods, output_path):
    '''
    name=name,
    marks_file="{}.json".format(name),
    methods=dataset['methods'],
    output_path=os.path.join('out2', name)

    Returns:
    '''
    # so we provide the thing with marks_file contains bounding boxes for all the detections
    with open(marks_file) as f: # it's fine
        data = json.load(f)

    os.mkdir(output_path)
    for method in methods:
        os.mkdir(os.path.join(out_folder, folder))
        for imname, rects in data.items():
            image = cv2.imread(in_file, 1)
            image = cv2.resize(image, (image_width, image_height))

            parts = []
            for color, (x, y, h, w) in zip(colors, rects):
                part = image[y:(y + h), x:(x + w), :]
                part = cv2.resize(part, (part_size, part_size))
                parts.append(part)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)

            bottom = np.vstack(
                [parts[0], np.zeros((image_height - 2 * part_size, part_size, 3), dtype=np.uint8), parts[1]])
            image = np.hstack([image, bottom])
            image = cv2.rectangle(image, (image_width, 3), (image_width + part_size - 2, part_size - 3,),
                                  color=colors[0], thickness=3)
            image = cv2.rectangle(image, (image_width, image_height - part_size + 2,),
                                  (image_width + part_size - 3, image_height - 2,), color=colors[1], thickness=3)
            print("{}\t\t\t|\t *** {} ***".format(folder.upper(), os.path.basename(in_file)))

            cv2.imwrite(out_file, image)


if __name__ == '__main__':
    if os.path.exists('out2'):
        os.system('rm -rf out2')
    os.mkdir('out2')

    with open('images.json') as f:
        data = json.load(f)

    for name, dataset in data.items():
        #print('Dataset: {}'.format(key))
        #path = "/Users/franz/devel/deweather/data/test/{}/input".format(key)
        print(dataset)

        print("Processing {}".format(name))
        print("{} : {} ".format(dataset['marks_file'], dataset['processed_path']))
        process_dataset(name=name,
                        marks_file="{}.json".format(name),
                        methods=dataset['methods'],
                        output_path=os.path.join('out2', name))
