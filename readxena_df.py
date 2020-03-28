import os
import pickle
import pandas
import re
import csv
import gzip
import torch

xf = gzip.open("/ibex/scratch/projects/c2014/tcga/xena/tcga_RSEM_gene_tpm.gz", "rt")
df = pandas.read_csv(xf, sep='\t', header=0, index_col=0)

with open('/ibex/scratch/projects/c2014/tcga/xena/tpm_df.pickle', 'wb') as f:
    pickle.dump((df), f, pickle.HIGHEST_PROTOCOL)
