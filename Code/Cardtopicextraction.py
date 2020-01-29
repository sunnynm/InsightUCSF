import gensim
import time
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
print("--- %s seconds ---" % (time.time() - start_time))

import pandas as pd
keypd = pd.read_csv('./Cardiolgykeywords.csv', header = None)
keylist = list(keypd.values.flatten())
keylist = [x.lower() for x in keylist]
print(keylist)
finalwordlist = []
with open('../Data/2Cardiolgydistance.csv', mode='w', encoding='UTF-8', newline='') as file:
	ewrite = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	ewrite.writerow(['Phrase', 'CardVal', 'HeartVal'])
	for word in keylist:
		#http://www.albertauyeung.com/post/generating-ngrams-python/
		tokens = [token for token in word.split(" ") if token != ""]
		for gram in range(1, 4):
			if len(tokens) < gram:
				break
			ngrams = zip(*[tokens[i:] for i in range(gram)])
			phrasetotry = ["-".join(ngram) for ngram in ngrams]
			for phrase in phrasetotry:
				try:
					cval = model.similarity('cardiology', phrase)
				except:
					cval = 0
				if cval != 0:
					finalwordlist.append(phrase)
					hval = model.similarity('heart', phrase)
					ewrite.writerow([phrase, cval, hval])
print("--- %s seconds ---" % (time.time() - start_time))

import pickle
filesave = 'refinedcardtopiclist.pickle'
with open(finalwordlist, 'wb') as fp:
	pickle.dump(iddict , fp)
print("--- %s seconds ---" % (time.time() - start_time))
print("done")