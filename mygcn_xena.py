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
import pandas

# constants, config
BATCHSIZE = 10
VALBATCHSIZE = 10
FILTERS = 150
EPOCHS = 500
MASKCOUNT = 500 # number of masks; actual masks are MASKCOUNT * BATCHSIZE
PATIENCE = 20

tcgafile = open('/ibex/scratch/projects/c2014/tcga/all.pickle', 'rb')
_, network, gene2protein, node2id = pickle.load(tcgafile)
tcgafile = open('/ibex/scratch/projects/c2014/tcga/xena/tpm_all_df.pickle', 'rb')
_, df, df_quantile, df_row, df_col = pickle.load(tcgafile)

dataall = torch.stack( [torch.tensor(df.values), torch.tensor(df_quantile.values), torch.tensor(df_row.values), torch.tensor(df_col.values)] , dim=2)

df_quantile.index = list(map(lambda x: x.split(".")[0], df_quantile.index)) # fix the df_quantile naming bug

id2node = { v: k for k, v in node2id.items() }
protein2gene = { v: k for k, v in gene2protein.items() }

dataset = torch.zeros(network.num_nodes, len(df.columns), 4) # number samples x number genes x number features
count = 0
for idx, row in enumerate(df.index):
    if row in gene2protein and gene2protein[row] in node2id:
        row_num = node2id[gene2protein[row]]
        dataset[row_num] = dataall[idx]


dataset_plain = torch.transpose(dataset, 0, 1)

trainset = dataset_plain[:int((len(dataset_plain)+1)*.90)]
testset = dataset_plain[int(len(dataset_plain)*.90+1):int(len(dataset_plain)*.95)]
validationset = dataset_plain[int(len(dataset_plain)*.95+1):]

print("Training/testing/validation sets ready")

# define the model layout -> use custom GCN model

class Net(torch.nn.Module):
    def __init__(self, edge_index, num_nodes):
        super(Net, self).__init__()
        self.conv1 = MyGCNConv(4, FILTERS, edge_index, num_nodes, node_dim = 1)
        self.conv2 = MyGCNConv(FILTERS, FILTERS, edge_index, num_nodes, node_dim = 1)
        #self.conv3 = MyGCNConv(FILTERS, FILTERS, edge_index, num_nodes, node_dim = 1)
        self.conv3 = MyGCNConv(FILTERS, 4, edge_index, num_nodes, node_dim = 1 )

    def forward(self, data):
        x = data
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x)
        out = torch.sigmoid(x)
        return out


# generate masks; TODO: add validation (some nodes never seen?)


masks = []
for i in range(MASKCOUNT):
    trainmask = torch.zeros(BATCHSIZE, network.num_nodes).bool().random_(0, 20) # 95% true, 5% false
    testmask = torch.ones(BATCHSIZE, network.num_nodes).bool() ^ trainmask # xor
    trainmask = torch.stack([trainmask, trainmask, trainmask, trainmask], dim=2)
    testmask = torch.stack([testmask, testmask, testmask, testmask], dim=2)
    masks.append((trainmask, testmask))

print("Mask sets ready")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device " + str(device))
cpu = torch.device('cpu')


loader = DataLoader(trainset, batch_size = BATCHSIZE, pin_memory = True)
validationloader = DataLoader(validationset, batch_size = VALBATCHSIZE, pin_memory = True)
testloader = DataLoader(testset, batch_size = VALBATCHSIZE, pin_memory = True)

print("Loaders ready")

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

print("Starting training")

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
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device))

    # validate the model
    torch.cuda.empty_cache()
    validationloss = 0.0
    for valbatch in validationloader:
        valbatch = valbatch.to(device)
        vout = model(valbatch)
        validationloss += F.mse_loss(vout, valbatch).item()
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device))
    valid_losses.append(validationloss)

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
