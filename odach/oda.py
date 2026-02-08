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
        self.imsize = image.shape[3]  # width for horizontal flip
        return image.flip(1)
    
    def batch_augment(self, images):
        self.imsize = images.shape[3]  # width for horizontal flip
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.imsize - boxes[:, [2,0]]
        return boxes

    
class VerticalFlip(Base):
    def augment(self, image):
        self.imsize = image.shape[2]  # height for vertical flip
        return image.flip(2)
    
    def batch_augment(self, images):
        self.imsize = images.shape[2]  # height for vertical flip
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.imsize - boxes[:, [3,1]]
        return boxes

    
class Rotate90Left(Base):
    def augment(self, image):
        self.imsize = image.shape[2]  # height
        # Rotate 90 degrees left: transpose and flip
        return torch.flip(image.transpose(2, 3), dims=[2])

    def batch_augment(self, images):
        self.imsize = images.shape[2]  # height
        # Rotate 90 degrees left: transpose and flip
        return torch.flip(images.transpose(2, 3), dims=[2])

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.imsize - boxes[:, [3,1]]
        res_boxes[:, [1,3]] = boxes[:, [0,2]]
        return res_boxes


class Rotate90Right(Base):
    def augment(self, image):
        self.imsize = image.shape[2]  # height
        # Rotate 90 degrees right: transpose and flip
        return torch.flip(image.transpose(2, 3), dims=[3])

    def batch_augment(self, images):
        self.imsize = images.shape[2]  # height
        return torch.flip(images.transpose(2, 3), dims=[3])

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [1,3]] = self.imsize - boxes[:, [2,0]]
        res_boxes[:, [0,2]] = boxes[:, [3,1]]
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
        self.imsize = image.shape[2]  # height
        return F.interpolate(image, scale_factor=self.imscale).flip(2)
    def batch_augment(self, images):
        self.imsize = images.shape[2]  # height
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
        self.imsize = image.shape[3]  # width
        return F.interpolate(image, scale_factor=self.imscale).flip(1)
    def batch_augment(self, images):
        self.imsize = images.shape[3]  # width
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
    
from .nms import nms
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
    def __init__(self, model, tta, scale=None, nms="wbf", iou_thr=0.5, skip_box_thr=0.5, weights=None):
        if scale is None:
            scale = [1]
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
        # Handle both single tensor and list of tensors
        if isinstance(img, (list, tuple)):
            img = torch.stack(img)
        elif not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor or list/tuple of tensors")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            img = img.cuda()
        
        n = img.size()[0]
        boxes_batch = [[] for x in range(n)]
        scores_batch = [[] for x in range(n)]
        labels_batch = [[] for x in range(n)]

        # Helper function to convert tensor or numpy to numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.asarray(x)

        # Initialize empty lists for collecting results
        # TTA loop
        for tta in self.ttas:
            # gen img
            inf_img = tta.batch_augment(img.clone())
            results = self.model_inference(inf_img)
            # iter for batch
            for idx, result in enumerate(results):
                box = to_numpy(result["boxes"])
                box = tta.deaugment_boxes(box)
                # scale box to 0-1
                if np.max(box, initial=1) > 1:
                    box[:, 0] /= img.shape[3]
                    box[:, 2] /= img.shape[3]
                    box[:, 1] /= img.shape[2]
                    box[:, 3] /= img.shape[2]

                thresh = 0.01
                scores_np = to_numpy(result["scores"])
                labels_np = to_numpy(result["labels"])
                ind = scores_np > thresh
                boxes_batch[idx].append(box[ind])
                scores_batch[idx].append(scores_np[ind])
                labels_batch[idx].append(labels_np[ind])
        outputs = []

        ##convert to torchvision output style
        for idx, (single_boxes, single_scores, single_labels) in enumerate(
                zip(boxes_batch, scores_batch, labels_batch)):
            output = {}

            single_boxes, single_scores, single_labels = self.nms(single_boxes, single_scores,
                                                                  single_labels)

            single_boxes[:, 0] *= img.shape[3]
            single_boxes[:, 1] *= img.shape[2]
            single_boxes[:, 2] *= img.shape[3]
            single_boxes[:, 3] *= img.shape[2]

            output['boxes'] = torch.from_numpy(single_boxes)
            output['scores'] = torch.from_numpy(single_scores)
            output['labels'] = torch.from_numpy(single_labels)

            outputs.append(output)

        return outputs

# for use in EfficientDets
class wrap_effdet:
    def __init__(self, model, imsize=512, device=None):
        # imsize.. input size of the model
        self.model = model
        self.imsize = imsize
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def __call__(self, img, score_threshold=0.22):
        # inference #
        img_info = torch.tensor([1]*img.shape[0]).float().to(self.device)
        det = self.model(img, img_info)

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
                "labels": torch.from_numpy(np.ones_like(npscore[indexes])).to(self.device)
            })

        return predictions


# for use in YOLOv5 and newer YOLO models from Ultralytics
class wrap_yolo:
    """
    Wrapper for YOLOv5 and newer YOLO models from Ultralytics.
    Handles the output format from YOLO models and converts it to the expected format.
    
    Args:
        model: YOLO model instance (e.g., YOLO('yolov8n.pt'))
        imsize: Input image size for the model (default: 640)
        score_threshold: Confidence threshold for filtering detections (default: 0.25)
        iou_threshold: IoU threshold for NMS (default: 0.45)
    """
    def __init__(self, model, imsize=640, score_threshold=0.25, iou_threshold=0.45):
        self.model = model
        self.imsize = imsize
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
    
    def __call__(self, img, score_threshold=None):
        """
        Run inference on input images and return results in the expected format.
        
        Args:
            img: Input tensor of shape (batch_size, channels, height, width)
            score_threshold: Override the default score threshold if provided
            
        Returns:
            List of dictionaries with 'boxes', 'scores', and 'labels' keys
        """
        if score_threshold is None:
            score_threshold = self.score_threshold
            
        # Convert torch tensor to numpy for YOLO inference
        # YOLO expects numpy arrays or PIL images
        if isinstance(img, torch.Tensor):
            # Convert to numpy
            img_np = img.cpu().numpy()
            if img_np.shape[1] == 3:  # CHW format
                img_np = np.transpose(img_np, (0, 2, 3, 1))  # Convert to BHWC
            
            # Convert to uint8 if needed
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
        elif isinstance(img, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in img):
            # Handle list of tensors
            img_np = []
            for tensor in img:
                np_tensor = tensor.cpu().numpy()
                if np_tensor.shape[0] == 3:  # CHW format
                    np_tensor = np.transpose(np_tensor, (1, 2, 0))  # Convert to HWC
                
                # Convert to uint8 if needed
                if np_tensor.dtype != np.uint8:
                    if np_tensor.max() <= 1.0:
                        np_tensor = (np_tensor * 255).astype(np.uint8)
                    else:
                        np_tensor = np_tensor.astype(np.uint8)
                
                # Ensure image has valid dimensions
                if np_tensor.shape[0] > 0 and np_tensor.shape[1] > 0:
                    img_np.append(np_tensor)
                else:
                    # Skip invalid images
                    continue
            
            if not img_np:
                raise ValueError("No valid images found in batch")
            # Keep as list to handle different image sizes
        else:
            raise TypeError("Input must be a torch.Tensor or list/tuple of tensors")

        predictions = []

        # Process each image in the batch
        for i in range(len(img_np)):
            # Run YOLO inference with explicit size
            results = self.model(img_np[i], conf=score_threshold, iou=self.iou_threshold, verbose=False, imgsz=self.imsize)
            
            # Extract boxes, scores, and labels
            # YOLO results is a list, so we need to access results[0]
            if hasattr(results, 'boxes') and results.boxes is not None:
                # Newer YOLO versions (v8+)
                boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)
            elif hasattr(results[0], 'boxes') and results[0].boxes is not None:
                # Newer YOLO versions (v8+) - results is a list
                boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                scores = results[0].boxes.conf.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy().astype(int)
            elif hasattr(results, 'xyxy') and results.xyxy is not None:
                # YOLOv5 format
                detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
                if len(detections) > 0:
                    boxes = detections[:, :4]  # x1, y1, x2, y2
                    scores = detections[:, 4]  # confidence scores
                    labels = detections[:, 5].astype(int)  # class labels
                else:
                    boxes = np.empty((0, 4))
                    scores = np.empty(0)
                    labels = np.empty(0, dtype=int)
            elif hasattr(results[0], 'xyxy') and results[0].xyxy is not None:
                # YOLOv5 format - results is a list
                detections = results[0].xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
                if len(detections) > 0:
                    boxes = detections[:, :4]  # x1, y1, x2, y2
                    scores = detections[:, 4]  # confidence scores
                    labels = detections[:, 5].astype(int)  # class labels
                else:
                    boxes = np.empty((0, 4))
                    scores = np.empty(0)
                    labels = np.empty(0, dtype=int)
            else:
                # Fallback for other formats
                boxes = np.empty((0, 4))
                scores = np.empty(0)
                labels = np.empty(0, dtype=int)
            
            # Filter by score threshold
            if len(scores) > 0:
                mask = scores >= score_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
            
            # Normalize boxes to [0, 1] range for TTA processing
            if len(boxes) > 0:
                # Get original image dimensions
                orig_h, orig_w = img_np[i].shape[:2]

                # Normalize coordinates (ensure float type for division)
                boxes_normalized = boxes.astype(np.float64)
                boxes_normalized[:, [0, 2]] /= orig_w  # x coordinates
                boxes_normalized[:, [1, 3]] /= orig_h  # y coordinates

                # Clamp to [0, 1] range
                boxes_normalized = np.clip(boxes_normalized, 0, 1)
            else:
                boxes_normalized = np.empty((0, 4), dtype=np.float64)
            
            # Convert to torch tensors and create output format
            predictions.append({
                'boxes': torch.from_numpy(boxes_normalized),
                'scores': torch.from_numpy(scores),
                'labels': torch.from_numpy(labels)
            })
        
        return predictions
