import gensim
import time
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
print("--- %s seconds ---" % (time.time() - start_time))

ml_problem_dict = {
	"denoising": "imagequality",
	"de-noising": "imagequality",
	"image post-processing": "imagequality",
	"image quality": "imagequality",


	"classification": "classification",

	"prediction": "prediction",
	"predict": "prediction",

	" lstm ": "sequenceanalysis",
	" rnn ": "sequenceanalysis",
	"recurrent neural network": "sequenceanalysis",
	"long short-term memory": "sequenceanalysis",	

	"segmentation": "segmentation",
	"semantic segmentation": "segmentation",

	"cross-modal": "crossmodal",

	"diagnosis": "diagnosis",
	"diagnostic": "diagnosis",
	"distinguish between": "diagnosis",

	"biomarker": "novelbiomarkers",
	"signature": "novelbiomarkers",
	"biologic insight": "novelbiomarkers",

	"synthetic": "syntheticdata",

	"object tracking": "objecttracking",
	"tracking": "objecttracking",

	# "": "protocolingscheduling",

}
import pandas as pd
keylist = list(set(ml_problem_dict.keys()).union(set(ml_problem_dict.values())))
print(keylist)
finalwordlist = []
import csv
with open('./RefinedMLProblist.csv', mode='w', encoding='UTF-8', newline='') as file:
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
filesave = 'refinedMLprobtopiclist.pickle'
with open(filesave, 'wb') as fp:
	pickle.dump(list(set(finalwordlist)) , fp)
print("--- %s seconds ---" % (time.time() - start_time))
print("done")