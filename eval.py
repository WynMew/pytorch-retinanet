import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import os.path as osp
from torch.autograd import Variable
import time
from PIL import Image
import model
import cv2
import numpy as np

assert torch.__version__.split('.')[1] == '4' # pytorch 0.4.* only
My_CLASSES = (  # always index 0
    '2guw', 'fm2a', 'nb5e', 'tbhp', 'xz7m',
    '6ate', '6uxf', 'h8u5', 'n1lb'
 #   'eoqx' # neg
    )
img_size = 512
MyClassNum = len(My_CLASSES)
#print('Loading model..')

retinanet = torch.load('retinanet_9_2.pt')
retinanet.training = False

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

idx=[]

for line in open(osp.join('/home/wynmew/workspace/Data', 'Data20180911GT')):
    idx.append(('/home/wynmew/workspace/Data', line.strip()))

counter=0
for i in range(len(idx)):
    line = idx[i]
    #print(line)
    img=osp.join(line[0],line[1])
    imgfile=img.split()[0]
    label = img.split()[5]
    #label = img.split()[1]
    if int(label) < 9:
        print(counter)
        print('G:', imgfile, label)
        imgCV2 = cv2.imread(imgfile)
        imgPre = cv2.cvtColor(imgCV2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(imgPre)
        img_pil_r = img_pil.resize((img_size, img_size))
        inputs = preprocess(img_pil_r)
        inputs.unsqueeze_(0)
        #start = time.time()
        scores, classification, transformed_anchors = retinanet(Variable(inputs.cuda(), volatile=True))
        idxs = np.where(scores > 0.01)
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            pts = bbox.tolist()
            lbs = classification[idxs[0][j]].tolist()
            scs = scores[idxs[0][j]].tolist()
            print('p:', pts, lbs, scs)
            print('--')
        counter+=1
        #end = time.time()
        #print('\ntime:', end - start)
