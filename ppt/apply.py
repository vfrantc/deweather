import os
import cv2
import json
import numpy as np
from pprint import pprint

colors = ((0, 255, 0), (0, 0, 255))     # colors that we use to mark the detections
image_width, image_height = 768, 512    # the size of the image
part_size = image_height // 2           # the size of the zoomed thing

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

        os.makedirs(os.path.join(output_path, name))

    for method in methods:
        for imname, rects in data.items():
            p = '../data/{}/{}'.format(method, imname);
            if not os.path.exists(p):
                if p.endswith('.png'):
                    p = p.replace('.png', '.jpg')
                else:
                    p = p.replace('.jpg', '.png')
            image = cv2.imread(p, 1)
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
            print("{}\t\t\t|\t *** {} ***".format(method.upper(), os.path.basename(name)))
            cv2.imwrite(os.path.join(output_path, name, imname[:-4] + method.replace('/', '_') + '.png'), image)

if __name__ == '__main__':
    # if the out folder does not exist - create it
    if os.path.exists('out2'):
        os.system('rm -rf out2')
    os.mkdir('out2')

    # open the configuration of the images
    with open('images.json') as f:
        data = json.load(f)

    for name, dataset in data.items(): # go through all the names -> datasets
        print('Name: ', name)
        print('Dataset: ', dataset)
        print('Methods: ', dataset['methods'])
        # In this case I have a fixed dataset, for this dataset I have particular methods that already processed it
        # Each dataset has a special .json file
        process_dataset(name=name,
                        marks_file="{}.json".format(name),
                        methods=dataset['methods'],
                        output_path='out2/')
