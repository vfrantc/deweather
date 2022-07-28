import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
import matplotlib.pyplot as plt

from qcnn import QuaternionConv
from qcnn import get_r, get_i, get_j, get_k

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = QuaternionConv(8, channel, kernel_size * 3, stride=1, padding=4)
        # Activated layers!
        self.net1_convs = nn.Sequential(QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = QuaternionConv(channel, 8, kernel_size, stride=1, padding=1)

    def edge_compute(self, x):
        x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

        y = x.new(x.size())
        y.fill_(0)
        y[:,:,:,1:] += x_diffx
        y[:,:,:,:-1] += x_diffx
        y[:,:,1:,:] += x_diffy
        y[:,:,:-1,:] += x_diffy
        #y = torch.sum(y,1,keepdim=True)/3
        y /= 4
        return y

    def forward(self, input_im):
        b, c, h, w = input_im.shape
        real = torch.zeros((b, 1, h, w)).cuda()

        edges = self.edge_compute(input_im)
        edges = torch.cat((real, edges), dim=1)
        input_im = torch.cat((real, input_im), dim=1)

        input = torch.cat((get_r(edges), get_r(input_im), get_i(edges), get_i(input_im), get_j(edges), get_j(input_im), get_k(edges), get_k(input_im)), dim=1)
        feats0   = self.net1_conv0(input)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)

        b = outs[:, 0, :, :].unsqueeze(1)
        R        = torch.sigmoid(torch.cat((outs[:, 0, :, :].unsqueeze(1), outs[:, 2, :, :].unsqueeze(1), outs[:, 4, :, :].unsqueeze(1), outs[:, 6, :, :].unsqueeze(1)), dim=1))
        L        = torch.sigmoid(torch.cat((outs[:, 1, :, :].unsqueeze(1), outs[:, 3, :, :].unsqueeze(1), outs[:, 5, :, :].unsqueeze(1), outs[:, 7, :, :].unsqueeze(1)), dim=1))
        return R, L

def get_decom(trainable=True):
    net = DecomNet().cuda()
    ckpt_dict  = torch.load('qdecomp.tar') # , map_location=torch.device('cpu')
    net.load_state_dict(ckpt_dict)
    for p in net.parameters():
        p.requires_grad = trainable
    return net

def decom_image(image, net):
    test_low_img   = 2*cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 1
    test_low_img   = np.transpose(test_low_img, (2, 0, 1))
    input_low_test = np.expand_dims(test_low_img, axis=0)
    input_low_test = Variable(torch.FloatTensor(torch.from_numpy(input_low_test))).cuda()
    R_low, I_low   = net(input_low_test)
    R_low = np.clip(np.transpose(R_low.cpu().detach().numpy().squeeze(), (1, 2, 0)), 0, 1)
    I_low = np.clip(np.transpose(I_low.cpu().detach().numpy().squeeze(), (1, 2, 0)), 0, 1)
    return R_low[:, :, 1:], I_low[:, :, 1:]

if __name__ == '__main__':
    net = get_decom()
    FNAME = 'input/input/010.png'
    dehazed_image = cv2.imread(FNAME)
    reflectance, illumination = decom_image(dehazed_image)

    fig, axs = plt.subplots(2, figsize=(16, 8))
    axs[0].imshow(reflectance)
    axs[1].imshow(illumination, cmap='gray')