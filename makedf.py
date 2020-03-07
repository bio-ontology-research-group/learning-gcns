import networkx as nx
import torch_geometric.utils as util
import os
import pickle
import re
import csv
import torch
import gzip

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


network = util.from_networkx(G)


# Read the cancer data -> read in tensor of shape NumSamples x NumNodes x NumFeatures [NumFeatures == 1]
count = 0
dataset_plain = []
for i in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("/ibex/scratch/projects/c2014/tcga/")) for f in fn]:
    if re.search("FPKM-UQ.txt.gz$", str(i)):
        count += 1
        if count % 20 == 0:
            print(str(count) + " files read")
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

with open('/ibex/scratch/projects/c2014/tcga/all.pickle', 'wb') as f:
    pickle.dump((dataset_plain, network, gene2protein, node2id), f, pickle.HIGHEST_PROTOCOL)

