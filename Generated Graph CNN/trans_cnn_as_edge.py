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
    out = self.bn(out)
    return out


class Node_OP(nn.Module):
  def __init__(self, Node, inplanes, outplanes):
    super(Node_OP, self).__init__()
    self.is_input_node = Node.type == 0
    self.input_nums = len(Node.inputs)
    self.Node = Node
    self.id = Node.id 
    self.convs = nn.ModuleDict([])
    dist = {}
    for i in range(self.input_nums):
      #dist[str(self.id) + '_' + str(i)] = nn.Parameter(torch.ones(1))
      dist[str(self.id) + '_' + str(i)] = nn.Parameter(2*torch.rand(1) - 1)
      self.convs[str(self.id) + '_' + str(i)] = Triplet_unit(outplanes, outplanes, stride=1)
    if self.input_nums == 0:
      self.convs['input'] = Triplet_unit(inplanes, outplanes, stride=2)
    self.mean_weight = nn.ParameterDict(dist)
    self.sigmoid = nn.Sigmoid()


  def forward(self, *input):
    if self.input_nums > 1:
      out = self.sigmoid(self.mean_weight[list(self.mean_weight.keys())[0]]) * self.convs[list(self.convs.keys())[0]](input[0])
      for i in range(1, len(self.mean_weight.keys())):
        out = out + self.sigmoid(self.mean_weight[list(self.mean_weight.keys())[i]]) * self.convs[list(self.convs.keys())[i]](input[i])
    else:
      out = self.convs[list(self.convs.keys())[0]](input[0])
    return out
  
  def get_edge_weight(self):
    return self.mean_weight


class StageBlock(nn.Module):
  def __init__(self, graph, inplanes, outplanes):
    super(StageBlock, self).__init__()
    self.nodes, self.input_nodes, self.output_nodes = get_graph_info(graph)
    self.nodeop  = nn.ModuleList()
    for node in self.nodes:
      self.nodeop.append(Node_OP(node, inplanes, outplanes))

  def forward(self, x):
    results = {}
    for id in self.input_nodes:
      results[id] = self.nodeop[id](x)
    for id, node in enumerate(self.nodes):
      if id not in self.input_nodes:
        results[id] = self.nodeop[id](*[results[_id] for _id in node.inputs])
    result = results[self.output_nodes[0]]
    for idx, id in enumerate(self.output_nodes):
      if idx > 0:
        result = result + results[id]
    result = result / len(self.output_nodes)
    return result
  
  def get_all_edge_weight(self):
    edge_weight = []
    for node in self.nodeop:
      if node.input_nums > 1:
        edge_weight.append(list(node.mean_weight.items()))
    result = edge_weight[0]
    for i in range(1, len(edge_weight)):
      result.extend(edge_weight[i])
    return result 
  
  def sort_edge_weight(self, weights):
    def mycmp(a):
      return abs(float(a[1].data))
    return sorted(weights, key=mycmp)
  
  def get_sort_edge_weight(self):
    return self.sort_edge_weight(self.get_all_edge_weight())
  
  def get_edge_num(self):
    return len(self.get_all_edge_weight())





class Trans_CNN_edge(nn.Module):
  def __init__(self, cfg):
    super(Trans_CNN_edge, self).__init__()
    self.cfg = cfg
    c = cfg['channels']
    self.conv1 = depthwise_separable_conv_3x3(3, c, 1)
    self.bn1 = nn.BatchNorm2d(c)
    self.conv2 = Triplet_unit(c, c, 1)
    self.graph_arr = []

    graph = build_graph(cfg['nodes'], cfg)
    self.graph_arr.append(graph)
    save_graph(graph, './graph_{}_conv3.yaml'.format(str(cfg["type"])))
    self.conv3 = StageBlock(graph, c, c)

    graph = build_graph(cfg['nodes'], cfg)
    self.graph_arr.append(graph)
    save_graph(graph, './graph_{}_conv4.yaml'.format(str(cfg["type"])))
    self.conv4 = StageBlock(graph, c, c *2)

    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(c * 2, 1280, kernel_size=1)
    self.bn2 = nn.BatchNorm2d(1280)
    self.avgpool = nn.AvgPool2d(8)
    self.fc = nn.Linear(1280, 10)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

  def change_graph(self, conv, graph):
    sort_edge_weight = conv.get_sort_edge_weight()
    _min_weight = sort_edge_weight[0]
    min_label = _min_weight[0].split('_')
    min_label = [int(i) for i in min_label]
    remove_edge(graph, min_label[0], min_label[1])
    del conv.nodeop[min_label[0]].mean_weight[_min_weight[0]]
    del conv.nodeop[min_label[0]].convs[_min_weight[0]]
    for node in conv.nodes:
      if node.id == min_label[0]:
        node.inputs.remove(min_label[1])
    for node in conv.nodeop:
      if node.id == min_label[0]:
        node.input_nums -= 1
  def change_network(self):
    
    self.change_graph(self.conv3, self.graph_arr[0])
    self.change_graph(self.conv4, self.graph_arr[1])
    save_graph(self.graph_arr[0], './graph_{}_conv3.yaml'.format(str(self.cfg["type"])))
    save_graph(self.graph_arr[1], './graph_{}_conv4.yaml'.format(str(self.cfg["type"])))

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)

    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    #x = self.conv5(x)
    x = self.relu(x)
    x = self.conv(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x



# %%
'''
cfg = {"name":"CNN", "type":"random_CNN_convert1", "bn":1, "batch_size":128, "epoches":240, "lr":0.1, "momentum":0.9, "weight_decay":0.0005, 
"device":"cuda:3", "nodes":15, 'graph_model':'ER', 'K':4, 'P' : 1, 'M':None, 'seed':1, "channels":78,
"reduction":16,
"rewire_K": 4,
"rewire_p":0.6}
net = Trans_CNN(cfg)

net.change_graph(net.conv3, net.graph_arr[0])
net.change_graph(net.conv3, net.graph_arr[0])
net.change_graph(net.conv3, net.graph_arr[0])
net.change_graph(net.conv3, net.graph_arr[0])
net.change_graph(net.conv3, net.graph_arr[0])

print(net.conv3.get_all_edge_weight())
import matplotlib.pyplot as plt
import networkx as nx
pos = nx.layout.spring_layout(net.graph_arr[0])

nodes = nx.draw_networkx_nodes(net.graph_arr[0], pos,  node_color='blue')
edges = nx.draw_networkx_edges(net.graph_arr[0], pos,  arrowstyle='->',
                               arrowsize=10, 
                               edge_cmap=plt.cm.Blues, width=2)
ax = plt.gca()
ax.set_axis_off()
plt.show()
x = torch.randn(1,3,32,32)
print(net(x))
#%%
def mycmp(item1, item2):
  if item1[1].

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
