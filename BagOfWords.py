import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack, save_npz
import json
from tqdm import tqdm

def fit_transform(file_name, sep=',', saveto=None):
  '''
  Fetch line by line from file and build BoW matrix and vocabulary.
  '''
  with open(file_name) as file_in:
    # If nlines is not None, read first nlines.
    vocab = []
    for k,line in tqdm(enumerate(file_in)):
      # Get list of syllables (or any other tokens)
      tokens = line.rstrip('\n').lower().split(sep)
      # Get unique tokens (switch back to list to preserve order)
      current_features = list(set(tokens))
      newfeatures = list(set(current_features) - set(vocab))
      vocab = vocab + newfeatures # vocab stays unique
      if k == 0: # Create new np array for mat for first sentence.
        mat = lil_matrix(np.zeros((1,len(vocab))))
      else: # Add columns to the matrix. New row hasn't been added, so all new columns are zeros.
        newcolumns = lil_matrix(np.zeros((mat.shape[0], len(newfeatures))))
        mat = hstack((mat, newcolumns))
        # Add the new row
        mat = lil_matrix(vstack((mat, lil_matrix(np.zeros((1,mat.shape[1]))))))
      counts = Counter(tokens)
      for f in current_features:
        f_indx = vocab.index(f)
        mat[-1,f_indx] = counts[f]
  mat = csr_matrix(mat)
  print('Saving to files...')
  save_npz(file_name+'.npz', mat)
  with open(file_name+'.vocab', 'w') as f:
    for t in vocab:
      f.write(t)
      f.write('\n')
  print('Done')
  return mat,vocab