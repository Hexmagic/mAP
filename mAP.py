import os
from collections import defaultdict
from operator import getitem
from prettytable import PrettyTable

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
        self.table = PrettyTable(['类别','AP'])

    def run(self):
        gt_map, gt_counter = self.init_gt()
        detect_map = self.init_dection()
        self.calmAP(gt_map, detect_map, gt_counter)

    def calAP(self, rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #     range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #     range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
        This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap

    def iou(self, bb, bbgt):
        bi = [
            max(bb[0], bbgt[0]),
            max(bb[1], bbgt[1]),
            min(bb[2], bbgt[2]),
            min(bb[3], bbgt[3])
        ]
        iw = bi[2] - bi[0] + 1
        ih = bi[3] - bi[1] + 1
        if iw > 0 and ih > 0:
            # compute overlap (IoU) = area of intersection / area of union
            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (
                bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
            ov = iw * ih / ua
            return ov
        return -1

    def acc(self, lst):
        for i in range(1, len(lst)):
            lst[i] = lst[i - 1] + lst[i]
        return lst

    def calmAP(self, gt, dt, gt_counter):
        sumAp = []
        num_classes = len(self.gt_classes)
        for class_name in self.gt_classes:
            length = len(dt[class_name])
            fp = [0] * length
            tp = [0] * length
            for ix, detect in enumerate(dt[class_name]):
                filename = detect['filename']
                box = list(map(int, detect['box']))
                grounds = gt[class_name][filename]
                iou_max, obj_matched = -1, -1
                for ground in grounds:
                    bbgt = list(map(int, ground['box']))
                    iou = self.iou(box, bbgt)
                    if iou > iou_max:
                        iou_max = iou
                        obj_matched = ground
                if iou_max > self.thresh:
                    if obj_matched['used'] is False:
                        obj_matched['used'] = True
                        tp[ix] = 1
                    else:
                        fp[ix] = 1
                else:
                    fp[ix] = 1
            tp = self.acc(tp)
            fp = self.acc(fp)
            rec = []
            for ele in tp:
                rec.append(ele / gt_counter[class_name])
            prec = []
            for i, ele in enumerate(tp):
                prec.append(tp[i] / (tp[i] + fp[i]))
            ap = self.calAP(rec, prec)
            self.table.add_row([class_name,round(ap,2)])
            sumAp.append(ap)
        mAp = sum(sumAp) / num_classes
        self.table.add_row(['mAP',round(mAp,2)])
        print(self.table)

    def init_gt(self):
        gt_counter = defaultdict(int)
        gt_map = defaultdict(lambda: defaultdict(list))
        for filename in os.listdir(self.gt_path):
            filepath = os.path.join(self.gt_path, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    class_name, a, b, c, d = line.split()
                    gt_map[class_name][filename].append({
                        'box': [a, b, c, d],
                        'used': False,
                    })
                    gt_counter[class_name] += 1
        return gt_map, gt_counter

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
                                     key=lambda x: x['conf'],
                                     reverse=True)
        return detect_map


cal = calmAP('input/ground-truth', 'input/detection-results')
cal.run()
