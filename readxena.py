import os
import pickle
import re
import csv
import gzip
import torch

xf = gzip.open("/ibex/scratch/projects/c2014/tcga/xena/tcga_RSEM_gene_tpm.gz", "rt")
reader = csv.reader(xf, delimiter="\t")
first = True
count = 0
cmap = {} # maps TCGA sample ID to the COLUMN
rmap = {} # maps ENGG gene ID to the ROW
t = torch.tensor([], dtype=torch.float32)
maxlength = -1

for col in reader:
    if first:
        first = False
        maxlength = len(col)-1
        for a in range(1, len(col)):
            cmap[a-1] = col[a]
    else:
        sample = col[0].split(".")[0]
        rmap[count] = sample
        vals = torch.tensor(list(map(lambda x: float(x), col[1:])), dtype=torch.float32)
        t = torch.cat((t, vals))
        # print(t.shape)
        count += 1


t = torch.reshape(t, (maxlength, -1))
with open('/ibex/scratch/projects/c2014/tcga/xena/tpm.pickle', 'wb') as f:
    pickle.dump((t, cmap, rmap), f, pickle.HIGHEST_PROTOCOL)

