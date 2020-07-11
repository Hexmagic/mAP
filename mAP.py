from collections import defaultdict
import os
from operator import getitem


class calmAP(object):
    def __init__(self, gt_path, pred_path, thresh=0.5):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.thresh = thresh
        self.gt_classes = [
            'backpack', 'bed', 'book', 'bookcase', 'bottle', 'bowl',
            'cabinetry', 'chair', 'coffeetable', 'countertop', 'cup',
            'diningtable', 'doll', 'door', 'heater', 'nightstand', 'person',
            'pictureframe', 'pillow', 'pottedplant', 'remote', 'shelf', 'sink',
            'sofa', 'tap', 'tincan', 'tvmonitor', 'vase', 'wastecontainer',
            'windowblind'
        ]

    def __call__(self):
        gt_map = self.init_gt()
        detect_map = self.init_dection()
        self.calmAP(gt_map, detect_map)

    def calAP(self, tp, fp):
        pass

    def iou(self, box1, box2):
        pass

    def calmAP(self, gt, dt):
        for index, class_name in enumerate(self.gt_classes):
            length = len(dt[class_name])
            fp = [0] * length
            np = [0] * length
            for detect in dt[class_name]:
                filename = detect['filename']
                box = list(map(int, detect['box']))
                ground = gt[filename]

    def init_gt(self):
        gt_map = defaultdict(lambda: defaultdict(list))
        for filename in os.listdir(self.gt_path):
            filepath = os.path.join(self.gt_path, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    class_name, a, b, c, d = line.split()
                    gt_map[filename].append({
                        'class': class_name,
                        'box': [a, b, c, d],
                        'used': False,
                    })
        return gt_map

    def init_dection(self):
        detect_map = defaultdict(list)
        for filename in os.listdir(self.pred_path):
            filepath = os.path.join(self.pred_path, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    class_name, conf, a, b, c, d = line.split()
                    detect_map[class_name].append({
                        'box': [a, b, c, d],
                        'filename': filename,
                        'conf': conf
                    })
        for key in detect_map.keys():
            detect_map[key] = sorted(detect_map[key],
                                     key=getitem('conf'),
                                     reverse=True)
        return detect_map
