import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def tokenize(smiles,tokens):
    index_list = np.arange(len(tokens))
    final = []*len(smiles) 
    for smile in smiles:
        smile_index = []*len(smile)
        for c in smile:
            c_index = tokens.index(c)
            smile_index.append(index_list[c_index]+1)
        final.append(smile_index)
    return final

def categorize(in_smiles):
    size_array = []*len(in_smiles)
    for smile in in_smiles:
        size_array.append(len(smile))
    max_len = max(size_array)
    
    categorized_smiles = []*len(in_smiles)
    for smile in in_smiles:
        cat_array = []*max_len
        for token in smile:
            cat_row = [0]*(len(tokens))
            cat_row[token-1] = 1
            cat_array.append(cat_row)
        nrows = max_len - len(cat_array) 
        each_row = [0]*len(tokens)
        for i in range(nrows):
            cat_array.append(each_row)
        categorized_smiles.append(cat_array)
    return categorized_smiles


full_df = pd.read_csv("/tmp/full.csv")
full_df = full_df.drop_duplicates(subset = ["SMILES"])
full_smiles = full_df['SMILES'].tolist()
smiles = full_smiles[:1000]
tokens = set()
for s in smiles:
    tokens = tokens.union(set(c for c in s))
tokens = sorted(list(tokens))
cat_smiles = categorize(tokenized_smiles)
cat_smiles = [np.array(s) for s in cat_smiles]
shape = cat_smiles[0].shape
cat_smiles = [s.reshape(shape[0]*shape[1]) for s in cat_smiles]
final = np.array(cat_smiles)
