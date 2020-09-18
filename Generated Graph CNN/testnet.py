#%%
import torch.nn as nn
from utils import Node, get_graph_info, build_graph, save_graph, load_graph, remove_edge
import torch
import math
import os
import torch.nn.functional as F



class depthwise_separable_conv_3x3(nn.Module):
  def __init__(self, nin, nout, stride):
    super(depthwise_separable_conv_3x3, self).__init__()
    self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin)
    self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

  def forward(self, x):
    out = self.depthwise(x)
    out = self.pointwise(out)
    return out


class Triplet_unit(nn.Module):
  def __init__(self, inplanes, outplanes, stride=1):
    super(Triplet_unit, self).__init__()
    self.relu = nn.ReLU()
    self.conv = depthwise_separable_conv_3x3(inplanes, outplanes, stride)
    self.bn = nn.BatchNorm2d(outplanes)

  def forward(self, x):
    out = self.relu(x)
    out = self.conv(out)
    #out = self.bn(out)
    return out





class Test_CNN(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    c = cfg['channels']
    self.conv1 = depthwise_separable_conv_3x3(3, c, 1)
    self.bn1 = nn.BatchNorm2d(c)
    self.conv2 = Triplet_unit(c, c, 1)
    self.conv3 = Triplet_unit(c, c, 2)
    self.conv4 = Triplet_unit(c,  2 * c, 2)

    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(c * 2, 1280, kernel_size=1)
    self.bn2 = nn.BatchNorm2d(1280)
    self.avgpool = nn.AvgPool2d(8)
    self.fc = nn.Linear(1280, 10)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

 

  def forward(self, x):
    x = self.conv1(x)
    #x = self.bn1(x)

    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    #x = self.conv5(x)
    x = self.relu(x)
    x = self.conv(x)
    #x = self.bn2(x)
    x = self.relu(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x



# %%
'''
cfg = {"name":"CNN", "type":"random_CNN_convert1", "bn":1, "batch_size":128, "epoches":240, "lr":0.1, "momentum":0.9, "weight_decay":0.0005, 
"device":"cuda:3", "nodes":5, 'graph_model':'ER', 'K':4, 'P' : 1, 'M':None, 'seed':1, "channels":78,
"reduction":16,
"rewire_K": 4,
"rewire_p":0.6}
net = Trans_CNN(cfg)



x = torch.randn(1,3,32,32)
print(net(x))

# %%
print(net.conv3.nodeop[6].mean_weight)




# %%
graph = load_graph('./graph.yaml')

# %%
import matplotlib.pyplot as plt
import networkx as nx
pos = nx.layout.spring_layout(graph)

nodes = nx.draw_networkx_nodes(graph, pos,  node_color='blue')
edges = nx.draw_networkx_edges(graph, pos,  arrowstyle='->',
                               arrowsize=10, 
                               edge_cmap=plt.cm.Blues, width=2)
ax = plt.gca()
ax.set_axis_off()
plt.show()
'''
# %%
