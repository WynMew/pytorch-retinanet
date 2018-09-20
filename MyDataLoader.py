from __future__ import print_function
import os
import sys
import random
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import time
import math
from myutils.box import box_iou, box_clamp

class MyDataset(data.Dataset):
    '''Load image/labels/boxes from a list file.

    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        #print(fname)
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()  # use clone to avoid any potential change.
        labels = self.labels[idx].clone()
        #print(labels)

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        #print(labels)
        #time.sleep(1)
        return img, boxes, labels

    def __len__(self):
        return self.num_imgs


def random_crop(
        img, boxes, labels,
        min_scale=0.3,
        max_aspect_ratio=2.):
    '''Randomly crop a PIL image.

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      labels: (tensor) bounding box labels, sized [#obj,].
      min_scale: (float) minimal image width/height scale.
      max_aspect_ratio: (float) maximum width/height aspect ratio.

    Returns:
      img: (PIL.Image) cropped image.
      boxes: (tensor) object boxes.
      labels: (tensor) object labels.
    '''
    imw, imh = img.size
    params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
    for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(100):
            scale = random.uniform(min_scale, 1)
            aspect_ratio = random.uniform(
                max(1/max_aspect_ratio, scale*scale),
                min(max_aspect_ratio, 1/(scale*scale)))
            w = int(imw * scale * math.sqrt(aspect_ratio))
            h = int(imh * scale / math.sqrt(aspect_ratio))

            x = random.randrange(imw - w)
            y = random.randrange(imh - h)

            roi = torch.Tensor([[x,y,x+w,y+h]])
            #print('label size', len(labels.size()))
            if len(labels.size())>0:
                #print('non empty labels',labels)
                ious = box_iou(boxes, roi)
                #print('ious:', ious)
                if ious.min() >= min_iou:
                    params.append((x,y,w,h))
                    break
            else:
                #print('empty label',labels)
                ious = Variable(torch.FloatTensor([[1]]))
                #print('my ious:', ious)
                #time.sleep(1)
                break

    x,y,w,h = random.choice(params)
    img = img.crop((x,y,x+w,y+h))

    if len(labels.size()) > 0:
        center = (boxes[:,:2] + boxes[:,2:]) / 2
        mask = (center[:,0]>=x) & (center[:,0]<=x+w) \
             & (center[:,1]>=y) & (center[:,1]<=y+h)
        if mask.any():
            boxes = boxes[mask.nonzero().squeeze()] - torch.Tensor([x,y,x,y])
            boxes = box_clamp(boxes, 0,0,w,h)
            labels = labels[mask]
        else:
            boxes = torch.Tensor([[0,0,0,0]])
            labels = torch.LongTensor([0])
        return img, boxes, labels
    else:
        boxes = torch.Tensor([])
        labels = torch.LongTensor([])
        return img, boxes, labels

def random_flip(img, boxes):
    '''Randomly flip PIL image.

    If boxes is not None, flip boxes accordingly.

    Args:
      img: (PIL.Image) image to be flipped.
      boxes: (tensor) object boxes, sized [#obj,4].

    Returns:
      img: (PIL.Image) randomly flipped image.
      boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        if boxes is not None:
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
    return img, boxes

def resize(img, boxes, size, max_size=1000, random_interpolation=False):
    '''Resize the input PIL image to given size.

    If boxes is not None, resize boxes accordingly.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#obj,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
      random_interpolation: (bool) randomly choose a resize interpolation method.

    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.

    Example:
    >> img, boxes = resize(img, boxes, 600)  # resize shorter side to 600
    >> img, boxes = resize(img, boxes, (500,600))  # resize image size to (500,600)
    >> img, _ = resize(img, None, (500,600))  # resize image only
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h

    method = random.choice([
        Image.BOX,
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR]) if random_interpolation else Image.BILINEAR
    img = img.resize((ow,oh), method)
    if boxes is not None:
        boxes = boxes * torch.Tensor([sw,sh,sw,sh])
    return img, boxes
