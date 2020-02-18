import gensim
import time
import datetime
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)

import pandas as pd
df = pd.read_csv("./MutliKeywordDistFULL.csv") 
alpha = df.loc[(df['ManRefVal'] >= .66)]
beta = df.loc[(df['ManRefVal'] >= .66) & (df['ManRefVal'] != 1)]
keys = list(set(beta.Phrase.values))

words_ak = []
embeddings_ak = []
for word in keys:
    embeddings_ak.append(model[word])
    words_ak.append(word)
print("--- %s seconds ---" % (time.time() - start_time))
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# modified t-SNE visulization from link below
# link: https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d

tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings_ak)

def tsne_plot_2d(title, embeddings, words=[], a=1):
    plt.figure(figsize=(16, 9))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, alpha=a)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.title(title)
    plt.grid(True)
    plt.savefig("keyword_relations.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()

tsne_plot_2d('Relations Between Discovered Keywords', embeddings_ak_2d,words_ak, a= 0.7)
print("--- %s seconds ---" % (time.time() - start_time))