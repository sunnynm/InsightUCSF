import gensim
import time
import datetime
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

keys = ['Paris', 'Python', 'Sunday']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=5):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

import pickle 

filesave = 'word_clusters.pickle'
with open(filesave, 'wb') as fp:
    pickle.dump(word_clusters , fp)
print("--- %s seconds ---" % (time.time() - start_time))

filesave = 'embedding_clusters.pickle'
with open(filesave, 'wb') as fp:
    pickle.dump(embedding_clusters , fp)
print("--- %s seconds ---" % (time.time() - start_time))

from sklearn.manifold import TSNE
import numpy as np

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
print(n)
print(m)
print(k)
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

filesave = 'embeddings_en_2d.pickle'
with open(filesave, 'wb') as fp:
    pickle.dump(embeddings_en_2d , fp)
print("--- %s seconds ---" % (time.time() - start_time))