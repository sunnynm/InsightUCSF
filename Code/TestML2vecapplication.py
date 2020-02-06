import gensim
import time
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
print("--- %s seconds ---" % (time.time() - start_time))


import pickle
filesave = './refinedMLtopiclist.pickle'
with open(filesave, 'rb') as fp:
    reflist = pickle.load(fp)

import pickle
filesave = './manuallyrefinedMLtopiclist.pickle'
with open(filesave, 'rb') as fp:
    manreflist = pickle.load(fp)
refleftover = list(set(reflist)-set(manreflist))

from rake_nltk import Rake #https://pypi.org/project/rake-nltk/
from nltk.corpus import stopwords
from nltk import word_tokenize 
from nltk.util import ngrams 
r = Rake(stopwords = stopwords.words("english")) # Uses stopwords for english from NLTK, and all puntuation characters.Please note that "hello" is not included in the list of stopwords.
counter = 0
import csv
import re
import pandas as pd
subdf = pd.read_csv('./subdf.csv')
with open('./MLKeywordDist.csv', mode='w', encoding='UTF-8', newline='') as file:
	ewrite = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	ewrite.writerow(['Document', 'Phrase', 'MLVal', 'RefinedWord', 'RefinedVal', "ManRefWord", "ManRefVal"])
	for index, row in subdf.iterrows():
		rtext = re.sub('[0-9]+', 'XXX', row['Abstract'])
		a = r.extract_keywords_from_text(rtext)
		c=r.get_ranked_phrases_with_scores()
		for j in c:
			if '©' in j[1]:
				continue
			cosinevallist = []
			tokens = [token for token in j[1].split(" ") if token != ""]
			for gram in range(1, 4):
				if len(tokens) < gram:
					break
				ngrams = zip(*[tokens[i:] for i in range(gram)])
				phrasetotry = ["-".join(ngram) for ngram in ngrams]
				for phrase in phrasetotry:
					try:
						cval = model.similarity('machine-learning', phrase)
						cword = 'machine-learning'
					except:
						cval = 0
					
					if cval != 0:
						manrefval = cval
						manrefword = cword
						for reference in manreflist:
							tempval = model.similarity(reference, phrase)
							if tempval > manrefval:
								manrefval = tempval
								manrefword = reference
						refval = manrefval
						refword = manrefword
						for reference in refleftover:
							tempval = model.similarity(reference, phrase)
							if tempval > refval:
								refval = tempval
								refword = reference
						ewrite.writerow([row["Title"], phrase, cval, refword, refval, manrefword, manrefval])
			#print([counter, j[1], j[0], mcval])
		if (counter % 20 == 0):
			print("--- %s seconds ---" % (time.time() - start_time))
			print(counter)
		counter+=1
print("CSV done")
print("--- %s seconds ---" % (time.time() - start_time))


print("--- %s seconds ---" % (time.time() - start_time))