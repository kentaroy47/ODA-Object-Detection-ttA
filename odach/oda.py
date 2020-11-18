# Written by @kentaroy47

# based on https://github.com/qubvel/ttach, https://github.com/andrewekhalel/edafa/tree/master/edafa and https://www.kaggle.com/shonenkov/wbf-over-tta-single-model-efficientdet

import torch
import torch.nn.functional as F
import numpy as np

class Base():
    def augment(self, image):
        # pass torch tensors
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class HorizontalFlip(Base):
    def augment(self, image):
        self.imsize = image.shape[1]
        return image.flip(1)
    
    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.imsize - boxes[:, [3,1]]
        return boxes

class VerticalFlip(Base):
    def augment(self, image):
        self.imsize = image.shape[1]
        return image.flip(2)
    
    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.imsize - boxes[:, [2,0]]
        return boxes
    
class Rotate90(Base):
    def augment(self, image):
        self.imsize = image.shape[1]
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.imsize - boxes[:, [3,1]] 
        res_boxes[:, [1,3]] = boxes[:, [0,2]]
        return res_boxes

class Multiply(Base):
    # change brightness of image
    def __init__(self, scale):
        # scale is a float value 0.5~1.5
        self.scale = scale
    def augment(self, image):
        return image*self.scale    
    def batch_augment(self, images):
        return images*self.scale    
    def deaugment_boxes(self, boxes):
        return boxes

class MultiScale(Base):
    # change scale of the image for TTA.
    def __init__(self, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale     
    def augment(self, image):
        return F.interpolate(image, scale_factor=self.imscale)  
    def batch_augment(self, images):
        return F.interpolate(images, scale_factor=self.imscale)
    def deaugment_boxes(self, boxes):
        return boxes/self.imscale

class MultiScaleFlip(Base):
    # change scale of the image and hflip.
    def __init__(self, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale
    def augment(self, image):
        self.imsize = image.shape[1]
        return F.interpolate(image, scale_factor=self.imscale).flip(2)
    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return F.interpolate(images, scale_factor=self.imscale).flip(3)
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.imsize*self.imscale - boxes[:, [2,0]]
        boxes = boxes/self.imscale
        return boxes

class MultiScaleHFlip(Base):
    # change scale of the image and vflip.
    # not useful for 2d detectors..
    def __init__(self, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale
    def augment(self, image):
        self.imsize = image.shape[1]
        return F.interpolate(image, scale_factor=self.imscale).flip(1)
    def batch_augment(self, images):
        self.imsize = images.shape[2]
        return F.interpolate(images, scale_factor=self.imscale).flip(2)
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.imsize*self.imscale - boxes[:, [2,0]]
        boxes = boxes/self.imscale
        return boxes
    
class TTACompose(Base):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)
    
from .nms import nms, soft_nms
from .wbf import weighted_boxes_fusion

class nms_func():
    """
    class to call nms during inference.
    """
    def __init__(self, nmsname="wbf", weights=None, iou_thr=0.5, skip_box_thr=0.1):
        self.weights = weights
        self.iou = iou_thr
        self.skip = skip_box_thr
        self.nms = nmsname
        
    def __call__(self, boxes_list, scores_list, labels_list):
        if self.nms == "wbf":
            return weighted_boxes_fusion(boxes_list, scores_list, labels_list, self.weights, self.iou, self.skip)
        elif self.nms == "nms":
            return nms(boxes_list, scores_list, labels_list, iou_thr=self.iou, weights=self.weights)
        # TODO: add soft-nms
        else:
            raise NotImplementedError()    

# Model wrapper
class TTAWrapper:
    """
    wrapper for tta and inference.
    model: your detector. Right now, must output similar to the torchvision frcnn model.
    mono: tta which do not configure the image size.
    multi: tta which configures the image size.
    These two must be declared separetly.
    nms: choose what nms algorithm to run. right now, wbf or nms.
    iou_thr: iou threshold for nms
    skip_box_thr: score threshold for nms
    weights: for weighted box fusion, but None is fine.
    """
    def __init__(self, model, tta, scale=[1], nms="wbf", iou_thr=0.5, skip_box_thr=0.5, weights=None):
        self.ttas = self.generate_TTA(tta, scale)
        self.model = model #.eval()       
        # set nms function
        # default is weighted box fusion.
        self.nms = nms_func(nms, weights, iou_thr, skip_box_thr)
    
    def generate_TTA(self, tta, scale):
        from itertools import product
        tta_transforms = []

        # Generate ttas for monoscale TTAs
        if len(scale)==1 and scale[0]==1:
            print("preparing tta for monoscale..")
            for tta_combination in product(*list([i, None] for i in tta)):
                tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
        # Multiscale TTAs
        else:
            print("preparing tta for multiscale..")
            for s in scale:
                for tta_combination in product(*list([i, None] for i in tta)):
                    tta_transforms.append(TTACompose([MultiScale(s)]
                                                     +[tta_transform for tta_transform in tta_combination if tta_transform]))
        return tta_transforms
    
    def model_inference(self, img):
        with torch.no_grad():
            results = self.model(img)
        return results
    
    def tta_num(self):
        return len(self.ttas)
    
    # TODO: change to call
    def __call__(self, img):
        boxes = []; scores = []; labels = [];
        # TTA loop
        for tta in self.ttas:
            # gen img
            inf_img = tta.batch_augment(img.clone())
            results = self.model_inference(inf_img)
            # iter for batch
            for result in results:
                box = result["boxes"].cpu().numpy()
                box = tta.deaugment_boxes(box)
                # scale box to 0-1
                if np.max(box)>1:
                    box[:,0] /= img.shape[3]
                    box[:,2] /= img.shape[3]
                    box[:,1] /= img.shape[2]
                    box[:,3] /= img.shape[2]

                thresh = 0.01
                ind = result["scores"].cpu().numpy() > thresh
                boxes.append(box[ind])
                scores.append(result["scores"].cpu().numpy()[ind])
                labels.append(result["labels"].cpu().numpy()[ind])

        boxes, scores, labels = self.nms(boxes, scores, labels)
        return boxes, scores, labels

# for use in EfficientDets
class wrap_effdet:
    def __init__(self, model, imsize=512):
        # imsize.. input size of the model
        self.model = model
        self.imsize = imsize
    
    def __call__(self, img, score_threshold=0.22):       
        # inference
        det = self.model(img, torch.tensor([1]*images.shape[0]).float().cuda())
        
        predictions = []
        for i in range(img.shape[0]):
            # unwrap output
            boxes = det[i][:,:4]  
            scores = det[i][:,4]
            # filter output
            npscore = scores.detach().cpu().numpy()
            indexes = np.where(npscore > score_threshold)[0]
            boxes = boxes[indexes]
            # coco2pascal
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            # clamp boxes
            boxes = boxes.clamp(0, self.imsize-1)
            # wrap outputs
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
                # TODO: update for multi-label tasks
                "labels": torch.from_numpy(np.ones_like(npscore[indexes])).cuda()
            })
            
        return predictions
