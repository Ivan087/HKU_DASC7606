from pycocotools.cocoeval import COCOeval
import json
import torch
from tqdm import tqdm

class Evaluation:
    def __init__(self):
        super(Evaluation, self).__init__()
        self.result = None

    def evaluate(self, dataset, model, threshold=0.05):
    
        model.eval()
        
        with torch.no_grad():

            # start collecting results
            results = []
            image_ids = []

            for index in tqdm(range(len(dataset))):
                data = dataset[index]
                scale = data['scale']

                # run network
                if torch.cuda.is_available():
                    scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                else:
                    scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
                scores = scores.cpu()
                labels = labels.cpu()
                boxes  = boxes.cpu()

                # correct boxes for image scale
                boxes /= scale

                if boxes.shape[0] > 0:
                    # change to (x, y, w, h) (MS COCO standard)
                    boxes[:, 2] -= boxes[:, 0]
                    boxes[:, 3] -= boxes[:, 1]

                    # compute predicted labels and scores
                    # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        # scores are sorted, so we can break
                        if score < threshold:
                            break

                        # append detection for each positively labeled class
                        image_result = {
                            'image_id'    : dataset.image_ids[index],
                            'category_id' : dataset.label_to_coco_label(label),
                            'score'       : float(score),
                            'bbox'        : box.tolist(),
                        }

                        # append detection to results
                        results.append(image_result)

                # append image to list of processed images
                image_ids.append(dataset.image_ids[index])

            if not len(results):
                return

            # write output
            json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

            # load results in COCO evaluation tool
            coco_true = dataset.coco
            coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

            # run COCO evaluation
            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAP_result = coco_eval.stats
            # print('map_result{}'.format(mAP_result))
            # self.result = coco_eval.stats
            # logger.info(mAP_result)
            # print('type of coco_eval.stat: {}'.format(type(coco_eval.stat)))
            # print(coco_eval.stat[0])
            # print('coco_eval{}'.format(coco_eval))
            # print('type of coco_eval.items(): {}'.format(type(coco_eval.items())))
            # print(coco_eval.items())
            # print(coco_eval.summarize(1))
            # for category, mAP in coco_eval.summary():
            #     print('map{}    cat{}'.format(mAP,category))
            #     writer.add_scalar(category, mAP)
            model.train()
        # mAP_result = coco_eval.stats
        # print('return {}'.format(mAP_result))

            return mAP_result
            # return 
