import torch
from transweather_model import Transweather

def get_dehaze(trainable=False):
  dehaze_net = Transweather()
  dehaze_net = dehaze_net.cuda()
  for param in dehaze_net.parameters():
      param.requires_grad = trainable
  dehaze_net.load_state_dict(torch.load('./trained/best'))
  return dehaze_net