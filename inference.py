import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchinfo import summary
import network
from datasets import Cityscapes
import numpy as np

# Loading pretrained model
model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=8)
model.load_state_dict( torch.load( Path('models', 'best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar') )['model_state']  )
sd = model.state_dict()
# print(sd.keys())
print(sd['backbone.layer3.7.bn1.running_mean'])
sd['backbone.layer3.7.bn1.running_mean'] = np.random.normal(loc=sd['backbone.layer3.7.bn1.running_mean'], scale=0.2)
print(sd['backbone.layer3.7.bn1.running_mean'])
# summary(model, (1,3,500,500))

# # Visualizing segmentation output
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()
# model.classifier.aspp.project[3].train()

# image_path = str(Path('/', 'datasets','cityscapes', 'leftImg8bit', 'train', 'aachen', 'aachen_000000_000019_leftImg8bit.png'))
# image = cv2.imread(image_path)

# downscale_ratio = 100
# width = int(downscale_ratio*image.shape[1]/100)
# height = int(downscale_ratio*image.shape[0]/100)
# image = cv2.resize(image, (width,height))

# transform = transforms.Compose([transforms.ToTensor()])
# image = transform(image).to(device)
# image = torch.reshape(image, (1,image.shape[0],image.shape[1],image.shape[2]))

# output = model(image)
# print(output[0,:, 300, 1024])
# pred = output.max(1)[1].detach().cpu().numpy()
# print(pred[0, 300, 1024])
# colorized_pred = Cityscapes.decode_target(pred).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# plt.imshow(colorized_pred[0])
# plt.savefig('./test_images/test.png')
