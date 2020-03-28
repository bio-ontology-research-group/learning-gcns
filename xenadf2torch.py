import os
import pickle
import pandas
import re
import csv
import gzip
import torch

with open('/ibex/scratch/projects/c2014/tcga/xena/tpm_df.pickle', 'rb') as f:
    df = pickle.load(f)
    print(df.index)
