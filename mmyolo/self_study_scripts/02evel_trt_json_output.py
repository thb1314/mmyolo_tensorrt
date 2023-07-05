import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import mmyolo
import os


mmyolo_dir = os.path.dirname(os.path.dirname(mmyolo.__file__))
coco_root = os.path.join(mmyolo_dir, 'data', 'coco')
image_root__dir= os.path.join(coco_root, 'val2017')

coco_gt = COCO(os.path.join(coco_root, 'annotations', 'instances_val2017.json'))

# get all coco class labels
coco_classes = [(v["id"], v["name"]) for k, v in coco_gt.cats.items()]
print("number of coco_classes: {}".format(len(coco_classes)))
coco_basename2imgid = {img_info['file_name']:img_id for img_id, img_info in coco_gt.imgs.items()}

# evaluate pytorch 
# coco_pred = coco_gt.loadRes(os.path.join(mmyolo_dir, 'self_study_scripts', 'yolov6_n_coco_result.bbox.json'))
# coco_evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType="bbox")
# coco_evaluator.evaluate()
# coco_evaluator.accumulate()
# coco_evaluator.summarize()

# evaluate tensort result 
annotations = []
with open(os.path.join(mmyolo_dir, 'self_study_scripts', 'dynamic_batch_trt_result.json'), 'r', encoding='utf-8') as fp:
    tensorrt_result_dict = json.load(fp)
for filepath, info in tensorrt_result_dict.items():
    basename = os.path.basename(filepath)
    img_id = coco_basename2imgid[basename]
    scores = info['scores']
    bboxes = info['bboxes']
    labels = info['labels']

    for score, box, label in zip(scores, bboxes, labels):
        cat_id = coco_classes[int(label)][0]
        x1, y1, x2, y2 = box
        box_width = max(0, (x2 - x1))
        box_height = max(0, (y2 - y1))
        annotations.append({
            'bbox': [x1, y1, box_width, box_height],
            'category_id': cat_id,
            'image_id': img_id,
            'score': score
        })

coco_pred = coco_gt.loadRes(annotations)
coco_evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType="bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
