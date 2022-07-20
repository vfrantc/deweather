import argparse
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from mmdet.core import voc_classes
from pprint import pprint

categories = {1: "person",
              2: "bicycle",
              3: "car",
              4: "motorcycle",
              5: "airplane",
              6: "bus",
              7: "train",
              8: "truck",
              9: "others",
              10: "traffic light",
              11: "fire hydrant",
              13: "stop sign",
              14: "parking meter",
              15: "bench",
              16: "bird",
              17: "cat",
              18: "dog",
              19: "horse",
              20: "sheep",
              21: "cow",
              22: "elephant",
              23: "bear",
              24: "zebra",
              25: "giraffe",
              27: "backpack",
              28: "umbrella",
              31: "handbag",
              32: "tie",
              33: "suitcase",
              34: "frisbee",
              35: "skis",
              36: "snowboard",
              37: "sports ball",
              38: "kite",
              39: "baseball bat",
              40: "baseball glove",
              41: "skateboard",
              42: "surfboard",
              43: "tennis racket",
              44: "bottle",
              46: "wine glass",
              47: "cup",
              48: "fork",
              49: "knife",
              50: "spoon",
              51: "bowl",
              52: "banana",
              53: "apple",
              54: "sandwich",
              55: "orange",
              56: "broccoli",
              57: "carrot",
              58: "hot dog",
              59: "pizza",
              60: "donut",
              61: "cake",
              62: "chair",
              63: "couch",
              64: "potted plant",
              65: "bed",
              67: "dining table",
              70: "toilet",
              72: "tv",
              73: "laptop",
              74: "mouse",
              75: "remote",
              76: "keyboard",
              77: "cell phone",
              78: "microwave",
              79: "oven",
              80: "toaster",
              81: "sink",
              82: "refrigerator",
              84: "book",
              85: "clock",
              86: "vase",
              87: "scissors",
              88: "teddy bear",
              89: "hair drier",
              90: "toothbrush"}


label_ids = {name: i for i, name in categories.items()}

def parse_xml(args):
    xml_path, img_path = args # path to image and xml
    print('Processing {}'.format(xml_path))

    tree = ET.parse(xml_path) # tree for a single image
    root = tree.getroot()     # root of the tree
    size = root.find('size')  # size -> find the
    old_w = int(size.find('width').text) # width of the width
    old_h = int(size.find('height').text) # height of the
    if old_w == 0:
        old_w = 1
    if old_h == 0:
        old_h = 1


    if old_h > old_w:
        new_w = int(old_w * 640 / old_h)
        new_h = 640
    else:
        new_w = 640
        new_h = int(old_h * 640 / old_w)

    h_ratio = new_h / old_h
    w_ratio = new_w / old_w

    bboxes = [] # find bounding boxes
    labels = [] # labels
    bboxes_ignore = []
    labels_ignore = []

    # go trough all the objects in the
    for obj in root.findall('object'):
        name = obj.find('name').text # name of the object
        label = label_ids[name] # label for this particular object
        difficult = int(obj.find('difficult').text) # mark if this is a difficult case
        bnd_box = obj.find('bndbox') # bndbox
        bbox = [
            int(int(bnd_box.find('xmin').text) * w_ratio), # xmin
            int(int(bnd_box.find('ymin').text) * h_ratio), # ymin
            int(int(bnd_box.find('xmax').text) * w_ratio), #
            int(int(bnd_box.find('ymax').text) * h_ratio)  # ymax
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)

    # filename
    # width x height
    #
    annotation = {
        'filename': img_path,
        'width': new_w,
        'height': new_h,
        #'difficult': int(obj.find('difficult').text),
        'difficult': 0,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64), # thease are ides????
            'bboxes_ignore': bboxes_ignore.astype(np.float32), # this is an optional
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, split, out_file):
    annotations = []

    filelist = osp.join(devkit_path, f'ImageSets/Main/{split}.txt')
    if not osp.isfile(filelist):
        print(f'filelist does not exist: {filelist},',  f'skip  {split}')
        return

    img_names = mmcv.list_from_file(filelist) # all the image files
    xml_paths = [osp.join(devkit_path, f'Annotations/{img_name}.xml') for img_name in img_names]
    img_paths = [f'JPEGImages/{img_name}.png' for img_name in img_names]
    part_annotations = mmcv.track_progress(parse_xml, list(zip(xml_paths, img_paths)))
    annotations.extend(part_annotations)

    '''
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorbike",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    '''
    easy = {i: 0 for i in range(10)}
    difficult = {i: 0 for i in range(10)}
    for ann in annotations:
        for label in ann['ann']['labels']:
            if ann['difficult']:
                difficult[label] += 1
            else:
                easy[label] += 1

    print(f'easy: {easy}')
    print(f'difficult: {difficult}')
    # print(annotations[0])


    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmcv.dump(annotations, out_file)

    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in categories.items():
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('--devkit_path', default='/home/franz/derain/data/RIS', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', default='/home/franz/derain/data/RIS', help='output path')
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mmcv.mkdir_or_exist(out_dir)

    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'

    prefix = 'new'
    dataset_name = prefix + '_test'
    print(f'processing {dataset_name} ...')
    cvt_annotations(devkit_path, 'test', osp.join(out_dir, dataset_name + out_fmt))

if __name__ == '__main__':
    main()
