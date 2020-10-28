# based on https://github.com/andrewekhalel/edafa/tree/master/edafa
# and https://www.kaggle.com/shonenkov/wbf-over-tta-single-model-efficientdet

import torch.nn.functional as F

class Base():
    def __init__(self, imsize):
        self.imsize = imsize

    def augment(self, image):
        # pass torch tensors
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(Base):
    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.imsize - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(Base):
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.imsize - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(Base):
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.imsize - boxes[:, [3,1]] 
        res_boxes[:, [1,3]] = boxes[:, [0,2]]
        return res_boxes

class TTAScale(Base):
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

class TTAMultiScale(Base):
    # change brightness of image
    def __init__(self, original, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale
        self.original = original
    def augment(self, image):
        return F.interpolate(image, scale_factor=self.imscale)  
    def batch_augment(self, images):
        return F.interpolate(images, scale_factor=self.imscale)
    def deaugment_boxes(self, boxes):
        return boxes/self.imscale

class TTAMultiScaleFlip(Base):
    # change brightness of image
    def __init__(self, original, imscale):
        # scale is a float value 0.5~1.5
        self.imscale = imscale
        self.original = original
    def augment(self, image):
        return F.interpolate(image, scale_factor=self.imscale).flip(2)
    def batch_augment(self, images):
        return F.interpolate(images, scale_factor=self.imscale).flip(3)
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.original*self.imscale - boxes[:, [2,0]]
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