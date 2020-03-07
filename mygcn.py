import networkx as nx
import csv
import torch_geometric.utils as util
import gzip
import os
import re
import pickle
from random import shuffle, randint
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from gcn import MyGCNConv

# constants, config
BATCHSIZE = 20
VALBATCHSIZE = 10
FILTERS = 32
EPOCHS = 300
MASKCOUNT = 250

tcgafile = open('/ibex/scratch/projects/c2014/tcga/all.pickle', 'rb')
dataset_plain, network, gene2protein, node2id = pickle.load(tcgafile)


print("Normalizing data")
# normalize
maxval = torch.max(torch.stack(dataset_plain))
dataset_plain = [t / maxval for t in dataset_plain]

shuffle(dataset_plain)
trainset = dataset_plain[:int((len(dataset_plain)+1)*.90)]
testset = dataset_plain[int(len(dataset_plain)*.90+1):int(len(dataset_plain)*.98)]
validationset = dataset_plain[int(len(dataset_plain)*.98+1):]

# define the model layout -> use custom GCN model

class Net(torch.nn.Module):
    def __init__(self, edge_index, num_nodes):
        super(Net, self).__init__()
        self.conv1 = MyGCNConv(1, FILTERS, edge_index, num_nodes, node_dim = 1)
#        self.conv2 = MyGCNConv(2 * FILTERS, FILTERS, edge_index, num_nodes, node_dim = 1)
        self.conv2 = MyGCNConv(FILTERS, 1, edge_index, num_nodes, node_dim = 1 )

    def forward(self, data):
        x = data
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv3(x)
        out = torch.sigmoid(x)
        return out


# generate masks; TODO: add validation (some nodes never seen?)

masks = []
for i in range(MASKCOUNT):
    trainmask = torch.zeros(BATCHSIZE, network.num_nodes).bool().random_(0, 10) # 90% true, 10% false (0)
    testmask = torch.ones(BATCHSIZE, network.num_nodes).bool() ^ trainmask # xor
    masks.append((trainmask, testmask))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device " + str(device))


loader = DataLoader(trainset, batch_size = BATCHSIZE, pin_memory = True)
validationloader = DataLoader(validationset, batch_size = VALBATCHSIZE)
testloader = DataLoader(testset, batch_size = VALBATCHSIZE)

model = Net(network.edge_index.to(device), network.num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(EPOCHS):
    for batch in loader:
        batch = batch.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(batch)
        # chose a random mask from the set of masks
        #currentmask = masks[randint(0, MASKCOUNT-1)]
        currentmask = masks[0] # fix mask
        trainmask = currentmask[0] # the trainmask
        trainmask = trainmask[0:len(batch)]
        loss = F.mse_loss(out[trainmask], batch[trainmask])
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        validationloss = 0.0
        for valbatch in validationloader:
            valbatch = valbatch.to(device)
            vout = model(valbatch)
            validationloss += F.mse_loss(vout, valbatch)
        print("Epoch " + str(epoch) + ": " + str(validationloss))

