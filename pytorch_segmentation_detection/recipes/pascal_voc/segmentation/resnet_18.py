import sys, os
sys.path.insert(0, '../../../../vision/')
sys.path.append('../../../../../pytorch-segmentation-detection/')

# Use second GPU -pytorch-segmentation-detection- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torch.autograd import Variable
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated

import numpy as np


#img_path = 'demo_img_vittal.jpg'
img_path = 'orange1.jpg'
valid_transform = transforms.Compose(
                [
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
print('here')
img_not_preprocessed = Image.open(img_path).convert('RGB').resize((512, 512))

img = valid_transform(img_not_preprocessed)
print('here')
img = img.unsqueeze(0)

img = Variable(img)
#img = Variable(img.cuda())
print('here')
fcn = resnet_dilated.Resnet18_8s(num_classes=21)
print('here')
fcn.load_state_dict(torch.load('resnet_18_8s_59.pth'))
fcn.cuda()
fcn.eval()

res = fcn(img)

_, tmp = res.squeeze(0).max(0)

segmentation = tmp.data.cpu().numpy().squeeze()

plt.imshow(img_not_preprocessed)
plt.show()

plt.imshow(segmentation)
plt.show()