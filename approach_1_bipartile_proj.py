import csv
from collections import defaultdict
from tqdm import tqdm

# Read data from file and store in a dictionary
data = defaultdict(dict)
with open('data/user_artists.dat', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        user_id, artist_id, weight = row
        data[user_id][artist_id] = int(weight)

# Define function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

# Define function to perform bipartite projection
def bipartite_projection(data, nodes):
    projection = defaultdict(set)
    for node1 in nodes:
        for node2 in nodes:
            if node1 != node2:
                common_artists = set(data[node1].keys()) & set(data[node2].keys())
                if common_artists:
                    weight = sum(data[node1][artist] for artist in common_artists)
                    projection[node1].add(node2)
                    projection[node2].add(node1)
    return projection

# Perform bipartite projection on user nodes
user_nodes = set(data.keys())
user_projection = bipartite_projection(data, user_nodes)

# Generate scores for each pair of nodes in user projection
scores = []
for user1 in tqdm(user_nodes):
    for user2 in user_projection[user1]:
        jaccard = jaccard_similarity(set(data[user1].keys()), set(data[user2].keys()))
        bias = sum(data[user1].values()) / sum(data[user2].values())
        for artist in data[user2]:
            if artist not in data[user1]:
                scores.append([user1, artist, bias, jaccard, data[user2][artist]/sum(data[user2].values())])

# Export scores to CSV file
with open('scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['User ID', 'Artist ID', 'Bias', 'Jaccard Similarity', 'Weight'])
    writer.writerows(scores)