#%%
from bindsnet         import encoding
from bindsnet.network import Network, nodes, topology, monitors
from bindsnet.learning import PostPre
import numpy as np
import random 
import math 
import torch 
import matplotlib.pyplot as plt 
from gabor import gabor
from tqdm import tqdm
#%%
def poisson(
    datum: torch.Tensor, time: int, dt: float = 1.0, device="cpu", **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size, device=device)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size, device=device).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(time, *shape)

def gen_rate(one_cycle, time, A, sigma, neurons):
    cycles = time // one_cycle
    rates = []
    centers = []
    for i in range(10):
        centers.append(neurons * random.random())

    for i in range(cycles):
        index = int(10 * random.random())
        center = centers[index]
        temp = []
        for n in range(neurons):
            if center < neurons // 2:
                temp_rate = A * math.exp(-1 * min((n - center), (center + 500 - n)) **2 / (2 * sigma ** 2))
            else:
                temp_rate = A * math.exp(-1 * min(abs((n - center)), abs((500 + n - center))) **2 / (2 * sigma ** 2))
            if temp_rate < 1e-10:
                temp_rate = 0
            temp.append(temp_rate)
        rates.append(temp)
    return rates 

def gen_input_spikes(rates, one_cycle, time):
    cycles = time // one_cycle
    spikes = poisson(datum=torch.Tensor(rates[0]), time=one_cycle)
    for i in range(1, cycles):
        spikes = torch.cat([spikes, poisson(datum=torch.Tensor(rates[i]), time=one_cycle)])
    return spikes 

rates = gen_rate(100, 100000, 30, 10, 400)

spikes = gen_input_spikes(rates, 100, 100000).numpy()

#%%
def gen_gabors(): #generate 18 gabor filters as feedforward connection weight
    gabors = []
    thetas = [0, 3.14/6, 3.14/4, 3.14/3, 3.14/2]
    lambdas = [0.3, 0.5, 1, 1.5]
    for l in lambdas:
        for t in thetas:
            gabors.append(torch.Tensor(gabor(1, t, l, 0, 1)).flatten().unsqueeze(0))
            
    return gabors

def gen_exc_inh_weights():
    w = torch.zeros(18, 5)
    exclist = list(range(18))
    for i in range(5):
        temp = random.sample(exclist, 14)
        for t in temp:
            w[t][i] = 1
    return w

def gen_inh_exc_weights():
    w = torch.zeros(5, 18)
    exclist = list(range(18))
    for i in range(5):
        temp = random.sample(exclist, 11)
        for t in temp:
            w[i][t] = 1
    return -1 * w

def check_weights(w):
    result = torch.zeros(18, 18)
    for i in range(18):
        for j in range(18):
            if w[i][j] > 0.6 :
                result[i][j] = 1
            if w[j][i] > 0.6 :
                result[j][i] = 1
            if result[i][j] == 1 and result[j][i] == 1:
                result[i][j] = 2
                result[j][i] = 2
                print(i)
                print(j)
    return result

gabors = gen_gabors()

w_exc_inh = gen_exc_inh_weights()
w_inh_exc = gen_inh_exc_weights()
print(w_inh_exc)

input_exc_weights = []
for i in range(3):
    input_exc_weights.append(gabors[0])
for i in range(3):
    input_exc_weights.append(gabors[1])
for i in range(3):
    input_exc_weights.append(gabors[2])
for i in range(9):
    input_exc_weights.append(gabors[i + 9])

input_exc_weights = torch.cat(input_exc_weights)
print(input_exc_weights.T.shape)

exc_exc_w = 0.75 * torch.rand(18, 18)
for i in range(18):
    exc_exc_w[i][i] = 0
checked = check_weights(exc_exc_w)
plt.matshow(checked)
#%%

network = Network(dt=1.0)  # Instantiates network.
inputnodes = nodes.Input(400, traces=True, tc_trace=20.0)  # Input layer.
excnodes = nodes.LIFNodes(18,
            traces=True,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,)  # Layer of LIF neurons.
inhnodes = nodes.LIFNodes(5,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=-40.0,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,)
input_exc = topology.Connection(source=inputnodes, target=excnodes, w=input_exc_weights.T, update_rule=PostPre, wmin=0, wmax=3, nu=(1e-4, 1e-4))  # Connection from X to Y.
input_inh = topology.Connection(source=inputnodes, target=inhnodes, w=0.5 * torch.rand(400, 5))
exc_inh = topology.Connection(source=excnodes, target=inhnodes, w=w_exc_inh)
inh_exc = topology.Connection(source=inhnodes, target=excnodes, w=w_inh_exc)
exc_exc = topology.Connection(source=excnodes, target=excnodes, w=exc_exc_w, update_rule=PostPre, wmin=0, wmax=0.75, nu=(1e-6, 1e-6))


# Add everything to the network object.
network.add_layer(layer=inputnodes, name='inputnodes')
network.add_layer(layer=excnodes, name='excnodes')
network.add_layer(layer=inhnodes, name='inhnodes')
network.add_connection(connection=input_exc, source='inputnodes', target='excnodes')
network.add_connection(connection=input_inh, source='inputnodes', target='inhnodes')
network.add_connection(connection=exc_inh, source='excnodes', target='inhnodes')
network.add_connection(connection=inh_exc, source='inhnodes', target='excnodes')
network.add_connection(connection=exc_exc, source='excnodes', target='excnodes')
network.to('cuda') 
#%%
spikes = torch.Tensor(spikes).cuda()
inputs = {'inputnodes': spikes}
#%%
ws = []
for i in range(10):
    network.run(inputs={'inputnodes': spikes[i*10000: (i+1)*10000]}, time = 10000)

    w = network.connections[('excnodes', 'excnodes')].w.cpu().numpy()
    ws.append(w)
    print(i)
#%%
for w in ws:
    checked = check_weights(w)
    plt.matshow(checked)
#%%
w = network.connections[('excnodes', 'excnodes')].w.cpu().numpy()
print(w)

checked = check_weights(w)
plt.matshow(checked)
#%%
w1 = network.connections[('inputnodes', 'excnodes')].w.T[6].view(20,20)
plt.matshow(w1.cpu().numpy())

# %%
print(torch.__version__)
# %%
