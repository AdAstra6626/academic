<<<<<<< HEAD
#%%

from torch.utils.data import DataLoader,Dataset
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from lenet import lenet
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar

def Train(cfg):
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True)  

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    device = torch.device('cuda:0')
    
    net = lenet()
    #net = nn.DataParallel(net, device_ids=[1, 2, 3, 4])
    net.to(device)
    epoch = cfg["epoches"]
    lr = cfg["lr"]
    momentum = cfg["momentum"]
    wd = cfg["weight_decay"]
    if cfg["resume"]:
        net.load_state_dict(torch.load(cfg["resume_path"]))

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = MultiStepLR(optimizer, milestones=[80,160], gamma=0.1)
    #acc_f = open('./result/{}_cifar10_acc.txt'.format(str(cfg["type"])),'w')
    #loss_f = open('./result/{}_cifar10_loss.txt'.format(str(cfg["type"])),'w')
    #acct_f = open('./result/{}_cifar10_acct.txt'.format(str(cfg["type"])),'w')
    epoch_acc = []
    for i in range(epoch):
        net.train()
        acc_arr = []
        loss_arr = []
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for j, (data, label) in enumerate(trainloader):
            optimizer.zero_grad()
            out = net(data.float().to(device))
            loss = criterion(out, label.to(device))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data.to(device)).sum()
            progress_bar(j, len(trainloader), 'Epoch:%d/%d Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (i, epoch, sum_loss/(j+1), 100.*correct/total, correct, total))
            loss_arr.append(float(sum_loss / (j + 1)))
            acc_arr.append(float(100. * correct / total))
        '''
        acc_f.write(str(acc_arr))
        acc_f.write('\n')
        loss_f.write(str(loss_arr))
        loss_f.write('\n')
        acc_f.flush()
        loss_f.flush()
        '''
        with torch.no_grad():
            correct = 0
            total = 0
            for k,data in enumerate(testloader):
                net.eval()
                images, labels = data
                images, labels = images.float().to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        progress_bar(k, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))
        acc = 100. * correct / total
        epoch_acc.append(acc)
        '''
        if acc==max(epoch_acc):
            torch.save(net.state_dict(), "../saved/{}_cifar10_best.pth".format(str(cfg["type"])))
        torch.save(net.state_dict(), "../saved/{}_cifar10_latest.pth".format(str(cfg["type"])))
        acct_f.write(str(acc))
        acct_f.write('\n')    
        acct_f.flush()
        '''
        scheduler.step()


cfg = {"name":"SENet","type":"SENet29", "bn":1, "batch_size":128, "epoches":240, "lr":0.1, "momentum":0.9, "weight_decay":0.0001,
        "resume":False, "resume_path":None, "depth":None}

if __name__ == "__main__":
    Train(cfg)


# %%
=======
#%%

from torch.utils.data import DataLoader,Dataset
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from lenet import lenet
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar

def Train(cfg):
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True)  

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    device = torch.device('cuda:0')
    
    net = lenet()
    #net = nn.DataParallel(net, device_ids=[1, 2, 3, 4])
    net.to(device)
    epoch = cfg["epoches"]
    lr = cfg["lr"]
    momentum = cfg["momentum"]
    wd = cfg["weight_decay"]
    if cfg["resume"]:
        net.load_state_dict(torch.load(cfg["resume_path"]))

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = MultiStepLR(optimizer, milestones=[80,160], gamma=0.1)
    #acc_f = open('./result/{}_cifar10_acc.txt'.format(str(cfg["type"])),'w')
    #loss_f = open('./result/{}_cifar10_loss.txt'.format(str(cfg["type"])),'w')
    #acct_f = open('./result/{}_cifar10_acct.txt'.format(str(cfg["type"])),'w')
    epoch_acc = []
    for i in range(epoch):
        net.train()
        acc_arr = []
        loss_arr = []
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for j, (data, label) in enumerate(trainloader):
            optimizer.zero_grad()
            out = net(data.float().to(device))
            loss = criterion(out, label.to(device))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data.to(device)).sum()
            progress_bar(j, len(trainloader), 'Epoch:%d/%d Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (i, epoch, sum_loss/(j+1), 100.*correct/total, correct, total))
            loss_arr.append(float(sum_loss / (j + 1)))
            acc_arr.append(float(100. * correct / total))
        '''
        acc_f.write(str(acc_arr))
        acc_f.write('\n')
        loss_f.write(str(loss_arr))
        loss_f.write('\n')
        acc_f.flush()
        loss_f.flush()
        '''
        with torch.no_grad():
            correct = 0
            total = 0
            for k,data in enumerate(testloader):
                net.eval()
                images, labels = data
                images, labels = images.float().to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        progress_bar(k, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))
        acc = 100. * correct / total
        epoch_acc.append(acc)
        '''
        if acc==max(epoch_acc):
            torch.save(net.state_dict(), "../saved/{}_cifar10_best.pth".format(str(cfg["type"])))
        torch.save(net.state_dict(), "../saved/{}_cifar10_latest.pth".format(str(cfg["type"])))
        acct_f.write(str(acc))
        acct_f.write('\n')    
        acct_f.flush()
        '''
        scheduler.step()


cfg = {"name":"SENet","type":"SENet29", "bn":1, "batch_size":128, "epoches":240, "lr":0.1, "momentum":0.9, "weight_decay":0.0001,
        "resume":False, "resume_path":None, "depth":None}

if __name__ == "__main__":
    Train(cfg)


# %%
>>>>>>> 47b5ee5c43ec987156a21eff05cb7238061b8a26
