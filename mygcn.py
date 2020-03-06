import networkx as nx
import csv
import torch_geometric.utils as util
import gzip
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from gcn import MyGCNConv

# constants, config

BATCHSIZE = 10


# read protein interactions

G=nx.Graph()

node2id = {}
count = 0

file = open("string/9606.protein.actions.v11.0.txt")
reader = csv.reader(file, delimiter='\t')
firstline = True
for column in reader:
    if firstline:
        firstline = False
    else:
        n1 = -1
        n2 = -1
        if column[0] in node2id:
            n1 = node2id[column[0]]
        else:
            node2id[column[0]] = count
            n1 = count
            count += 1

        if column[1] in node2id:
            n2 = node2id[column[1]]
        else:
            node2id[column[1]] = count
            n2 = count
            count += 1

        G.add_node(n1)
        G.add_node(n2)
        score = int(column[6])
        if score>=700:
            G.add_edge(n1, n2)


# read map of gene ids to protein ids, map to nodes
file = open("string/9606.protein.aliases.v11.0.txt")
reader = csv.reader(file, delimiter='\t')
firstline = True
gene2protein = {}
for col in reader:
    if firstline:
        firstline = False
    else:
        prot = col[0]
        gene = col[1]
        if gene.find("ENSG")>-1 and prot in node2id:
            gene2protein[gene] = prot


# generate PyTorch datastructure from networkx

network = util.from_networkx(G)

# Read the cancer data -> read in tensor of shape NumSamples x NumNodes x NumFeatures [NumFeatures == 1]


dataset_plain = []
count = 0
for i in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("cancer/")) for f in fn]:
    if str(i).find("FPKM-UQ") > -1 and count < 20:
        count += 1
        file = gzip.open(i, mode="rt")
        csvobj = csv.reader(file, delimiter = '\t')
        localX = torch.zeros(network.num_nodes,1)
        for line in csvobj:
            gene = line[0].split(".")[0]
            if gene in gene2protein:
                protein = gene2protein[gene]
                nodeid = node2id[protein]
                evalue = float(line[1])
                #print(gene + " is not missing: " + protein + " with id " + str(nodeid))
                localX[nodeid] = evalue
            else:
                continue
        dataset_plain.append(localX)

# normalize
maxval = torch.max(torch.stack(dataset_plain))
dataset_plain = [t / maxval for t in dataset_plain]


# define the model layout -> use custom GCN model defined above

class Net(torch.nn.Module):
    def __init__(self, edge_index, num_nodes):
        super(Net, self).__init__()
        self.conv1 = MyGCNConv(1, 8, edge_index, num_nodes, node_dim = 1)
        self.conv2 = MyGCNConv(8, 1, edge_index, num_nodes, node_dim = 1 )

    def forward(self, data):
        x = data
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


# generate masks; TODO: add validation
    
trainmask = torch.zeros(BATCHSIZE, network.num_nodes).bool().random_(0, 10)
testmask = torch.ones(BATCHSIZE, network.num_nodes).bool()
validationmask = torch.ones(BATCHSIZE, network.num_nodes).bool()
for i in range(0, BATCHSIZE - 1):
    for j in range(0, network.num_nodes):
        if trainmask[i][j]:
            testmask[i][j] = False
        else:
            testmask[i][j] = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device " + str(device))
loader = DataLoader(dataset_plain, batch_size = BATCHSIZE, shuffle = True, pin_memory = True)

model = Net(network.edge_index, network.num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

for epoch in range(10):
    for batch in loader:
        batch = batch.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out[trainmask], batch[trainmask])
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        out = model(batch)
        testloss = F.mse_loss(out[testmask], batch[testmask])
        print("Epoch " + str(epoch) + ": " + str(testloss))

