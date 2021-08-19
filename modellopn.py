
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

class LOPN(nn.Module):
    """ (Pairwire frame concat) """
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(LOPN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.class_num=int(self.class_num/2)

        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        pair_num=pair_num
        self.fc8 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
      f = []  # frame features
      pf = []  # pairwise concat
 
      for ii in range(self.tuple_len):
          for j in range(ii+1, self.tuple_len):
              a = tuple[:, ii, :, :, :]
              b = tuple[:, j, :, :, :]
              clip=torch.stack([a, b], dim=2)
              pf.append(self.base_network(clip))

 
      pf = [self.fc7(i) for i in pf]
      pf = [self.relu(i) for i in pf]
      h = torch.cat(pf, dim=1)
      h = self.dropout(h)
      h = self.fc8(h)  # logits

      return h
