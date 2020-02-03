import gensim
import time
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
print("--- %s seconds ---" % (time.time() - start_time))
print(model.most_similar(positive=['cardiology']))
print("--- %s seconds ---" % (time.time() - start_time))
