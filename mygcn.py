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
import numpy as np
from earlystopping import EarlyStopping

# constants, config
BATCHSIZE = 20
VALBATCHSIZE = 10
FILTERS = 32
EPOCHS = 500
MASKCOUNT = 5000 # number of masks; actual masks are MASKCOUNT * BATCHSIZE
PATIENCE = 20

tcgafile = open('/ibex/scratch/projects/c2014/tcga/all.pickle', 'rb')
dataset_plain, network, gene2protein, node2id = pickle.load(tcgafile)


print("Normalizing data")
# normalize
maxval = torch.max(torch.stack(dataset_plain))
dataset_plain = [t / maxval for t in dataset_plain]

shuffle(dataset_plain)
trainset = dataset_plain[:int((len(dataset_plain)+1)*.90)]
testset = dataset_plain[int(len(dataset_plain)*.90+1):int(len(dataset_plain)*.95)]
validationset = dataset_plain[int(len(dataset_plain)*.95+1):]

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
    trainmask = torch.zeros(BATCHSIZE, network.num_nodes, 1).bool().random_(0, 10) # 90% true, 10% false
    testmask = torch.ones(BATCHSIZE, network.num_nodes, 1).bool() ^ trainmask # xor
    masks.append((trainmask, testmask))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device " + str(device))


loader = DataLoader(trainset, batch_size = BATCHSIZE, pin_memory = True)
validationloader = DataLoader(validationset, batch_size = VALBATCHSIZE)
testloader = DataLoader(testset, batch_size = VALBATCHSIZE)

model = Net(network.edge_index.to(device), network.num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = [] 
early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)


for epoch in range(EPOCHS):
    for batch in loader:
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad()
        trainmask, testmask = masks[randint(0, MASKCOUNT-1)] # chose a random mask from the set of masks, use for this batch
        trainmask = trainmask[0:len(batch)].to(device)
        testmask = testmask[0:len(batch)].to(device)
        batch *= trainmask # mask (set to zero) the features of the training nodes in this batch
        out = model(batch)
        loss = F.mse_loss(out[testmask], batch[testmask]) # compute loss over the nodes previously hidden
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # validate the model
    validationloss = 0.0
    for valbatch in validationloader:
        valbatch = valbatch.to(device)
        vout = model(valbatch)
        validationloss += F.mse_loss(vout, valbatch)
    valid_losses.append(validationloss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    epoch_len = len(str(EPOCHS))
    print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')
    print(print_msg)
    # clear lists to track next epoch
    train_losses = []
    valid_losses = []
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))
torch.save(model.state_dict(), 'tcga.pt')
