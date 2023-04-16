import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import csv
from tqdm import tqdm

# Read data from file and store in a pandas DataFrame
df = pd.read_csv('data/user_artists.dat', header=None, names=['user_id', 'artist_id', 'weight'])

# Convert data to sparse matrix
users = df['user_id'].unique()
artists = df['artist_id'].unique()
row = df['user_id'].astype('category').cat.codes
col = df['artist_id'].astype('category').cat.codes
data = df['weight']
adj_matrix = csr_matrix((data, (row, col)), shape=(len(users), len(artists)))


# Define functions for Jaccard similarity and bias calculation
def bias(u):
    return adj_matrix[u].sum()

def jaccard_similarity(u, v):
    # Find the set of all neighbors of u and v
    neighbors_u = set(adj_matrix[u].nonzero()[1])
    neighbors_v = set(adj_matrix[v].nonzero()[1])
    # Compute the intersection and union of the neighbor sets
    intersection = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)
    # Compute the sum of weights of edges incident on the neighbors of u and v
    neighbor_edge_weight_sum = adj_matrix[:, list(union)].sum(axis=0)
    # Compute the sum of weights of edges incident on the neighbors of u and v that are in the intersection set
    neighbor_intersection_weight_sum = adj_matrix[:, list(intersection)].sum(axis=0)
    # Compute the Jaccard similarity, accounting for edge weights
    jaccard = neighbor_intersection_weight_sum.sum() / neighbor_edge_weight_sum.sum()
    return jaccard

# Define function for computing scores
def compute_scores_and_write(u, v):
    with open('scores2.csv', 'a') as f:
        jaccard = jaccard_similarity(u, v)
        if jaccard > 0.05 and u != v:  # Only add results if jaccard is greater than 0.05
            bias_u = bias(u)
            bias_v = bias(v)
            scores = []
            for z in adj_matrix[v].nonzero()[1]:
                    weight = adj_matrix[v, z]
                    scores.append([users[u], artists[z], bias_u, bias_v, bias_u/bias_v, jaccard, weight/bias(v), adj_matrix[u,z]])
            # Write scores to file as they come
            for score in scores:
                f.write(','.join([str(x) for x in score]) + '\n')
            return scores
        else:
            return None

# Use parallel processing to compute scores for all pairs of nodes
with open('scores2.csv', 'w') as f:
    # Write header to file
    f.write('User ID,Artist ID,Bias_U,Bias_V,Bias_U/Bias_V,Jaccard Similarity,Normalised_Weight,Actual_Val\n')
with tqdm(total=len(users)*len(users)) as pbar:
    Parallel(n_jobs=-1)(delayed(compute_scores_and_write)(u, v) for u in range(len(users)) for v in adj_matrix[u].nonzero()[0])
    pbar.update(1)