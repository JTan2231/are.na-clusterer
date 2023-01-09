import nn
import json
import joblib
import requests

from math import sqrt

import numpy as np
from sklearn.cluster import KMeans

with open('list', 'r') as f:
    slugs = [slug.split('/')[-1] for slug in f.read().splitlines()]

print("slugs:")
for slug in slugs:
    print(f'- {slug}')

API_BASE = 'http://api.are.na/v2/'
PER = 250

blocks = []

for slug in slugs:
    response = requests.get(f'{API_BASE}channels/{slug}/?per={PER}')
    channel = json.loads(response.content)
    blocks += [(block['content'], block['id']) for block in channel['contents'] if block['class'] == 'Text']

embeddings_masks = [nn.create_entry_embedding(block[0]) for block in blocks]
embeddings = []
masks = []
for (embed, mask) in embeddings_masks:
    embeddings.append(embed)
    masks.append(mask)

score_matrix = np.vstack(embeddings)
score_matrix = np.mean(score_matrix, axis=1)
score_matrix = np.reshape(score_matrix, (len(embeddings), -1))

n = int(sqrt(PER))
kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42, n_init='auto')
kmeans.fit(score_matrix)
labels = kmeans.labels_

groups = { label: [] for label in labels}
for i in range(len(labels)):
    groups[labels[i]].append(blocks[i])

js_string = 'export const clusteredBlocks = ['
for k in groups.keys():
    js_string += '['
    for block in groups[k]:
        js_string += f'[`{block[0]}`,{block[1]}],\n'

    js_string += '],\n'

js_string += '];'

with open('blocks.js', 'w') as f:
    f.write(js_string)

print('finished')
