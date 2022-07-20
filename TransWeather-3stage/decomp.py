'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
import matplotlib.pyplot as plt
import kornia as K

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=5):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(15, channel, kernel_size * 3,  padding=7, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size, padding=2, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 6, kernel_size, padding=2, padding_mode='replicate')

    def forward(self, input_im):
        #input_img = torch.cat((input_im, input_im), dim=1)

        grads = K.filters.spatial_gradient(input_im, order=1)  # BxCx2xHxW
        grads1_x = grads[:, :, 0]
        grads1_y = grads[:, :, 1]
        grads = K.filters.spatial_gradient(input_im, order=2)  # BxCx2xHxW
        grads2_x = grads[:, :, 0]
        grads2_y = grads[:, :, 1]

        input = torch.cat((input_im, grads1_x, grads1_y, grads2_x, grads2_y), dim=1)

        feats0   = self.net1_conv0(input)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:6, :, :])
        return R, L

def get_decom(trainable=True):
  net = DecomNet().cuda()
  ckpt_dict  = torch.load('decomp.tar') # , map_location=torch.device('cpu')
  net.load_state_dict(ckpt_dict)
  for p in net.parameters():
      p.requires_grad = trainable
  return net

def decom_image(image, net):
  test_low_img   = 2*cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) - 1 / 255.0
  test_low_img   = np.transpose(test_low_img, (2, 0, 1))
  input_low_test = np.expand_dims(test_low_img, axis=0)
  input_low_test = Variable(torch.FloatTensor(torch.from_numpy(input_low_test))).cuda()
  R_low, I_low   = net(input_low_test)
  R_low = np.clip(np.transpose(R_low.cpu().detach().numpy().squeeze(), (1, 2, 0)), 0, 1)
  I_low = np.clip(np.transpose(I_low.cpu().detach().numpy().squeeze(), (1, 2, 0)), 0, 1)
  return R_low, I_low

if __name__ == '__main__':
    net = get_decom()
    FNAME = 'input/input/010.png'
    dehazed_image = cv2.imread(FNAME)
    reflectance, illumination = decom_image(dehazed_image)

    fig, axs = plt.subplots(2, figsize=(16, 8))
    axs[0].imshow(reflectance)
    axs[1].imshow(illumination, cmap='gray')
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
import matplotlib.pyplot as plt
import kornia as K

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0] # max value of R, G, B
        input_img= torch.cat((input_max, input_im), dim=1)     #
        feats0   = self.net1_conv0(input_img) # first convolution, # lear transform
        featss   = self.net1_convs(feats0)    #
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :]) # reflectance
        L        = torch.sigmoid(outs[:, 3:4, :, :]) # light
        L = torch.cat((L, L, L), dim=1)
        return R, L

def get_decom(trainable=True):
  net = DecomNet().cuda()
  ckpt_dict  = torch.load('9200.tar') # , map_location=torch.device('cpu')
  net.load_state_dict(ckpt_dict)
  for p in net.parameters():
      p.requires_grad = trainable
  return net
