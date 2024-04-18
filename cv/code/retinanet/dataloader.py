from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from torchvision import transforms
from torchvision.transforms import Grayscale, RandomGrayscale

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
from skimage import exposure

from PIL import Image, ImageEnhance


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))
        # print(annotations_ids)
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        ##################################################################
        # TODO: Please substitute the "?" to transform annotations
        #       from [x, y, w, h] to [x1, y1, x2, y2]
        ##################################################################
            

        annotations[:, 2] = annotations[:, 0] +  annotations[:, 2] # x2
        annotations[:, 3] = annotations[:, 1] +  annotations[:, 3] # y2

        ##################################################################
        
        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):

        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return len(self.coco.getCatIds())



def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]     # (B,nums,5) where 'nums' is the number of annotations for a single image, which differs among imgs
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors.
        The oupur image will have at least one size satisfied 608 or 1024. And both size can be divided by 32
    """

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """ Random Horizontal and Vertical Flip .
    Guassian noise. 
    """

    def __call__(self, sample, flip_x=0.5,noise_p = 0.2,std = 10):
        
        image, annots = sample['img'], sample['annot']
        
        rows, cols, channels = image.shape

        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

        # if np.random.rand() < flip_y:
        #     image = image[::-1, :, :]

        #     y1 = annots[:, 1].copy()
        #     y2 = annots[:, 3].copy()
            
        #     y_tmp = y1.copy()

        #     annots[:, 1] = rows - y2
        #     annots[:, 3] = rows - y_tmp
        
        #调节亮度
        # Color Jitter
        if np.random.rand() < 0.3:
            brightness = np.random.uniform(max(0, 1 - 0.2), 1 + 0.2)
            contrast = np.random.uniform(max(0, 1 - 0.2), 1 + 0.2)
            saturation = np.random.uniform(max(0, 1 - 0.2), 1 + 0.2)

            image = np.clip(image * brightness, 0, 255)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * contrast + mean, 0, 255)
            # Note: A more sophisticated color jitter might be applied here using other libraries like PIL or OpenCV

        # if np.random.rand() < 0.2:
        #     image = exposure.adjust_gamma(image, gamma=0.7,gain=1)
            # image_dark = exposure.adjust_gamma(image, gamma=1.5,gain=1)
        # if np.random.rand() < 0.1:
        #     image = Image.fromarray(image, mode='RGB')
        #     image = Grayscale(num_output_channels=3)(image)
        #     image = np.asarray(image)

        if np.random.rand() < noise_p:
            noise = np.random.normal(loc=0, scale=std, size=(rows, cols, channels))
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        

        if np.random.rand() < 0.1:
            
            grayscale_weights = np.array([0.2989, 0.5870, 0.1140])
            gray_img = np.dot(image, grayscale_weights)
            gray_img_stacked = np.stack([gray_img, gray_img, gray_img], axis=-1)
            image = gray_img_stacked

        sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    """
    (H,W,C)
    """

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]]) # (1,1,3)
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        ###################################################################
        # TODO: Please modify and fill the codes here to complete the image normalization
        ##################################################################
        image = image.astype(np.float32)
        image = (image - self.mean)/self.std

        ##################################################################

        return {'img':(image), 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
