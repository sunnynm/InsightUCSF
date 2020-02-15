import gensim
import time
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
print("--- %s seconds ---" % (time.time() - start_time))

data_modality_dict = {

	" ehr ": "ehr",
	" emr ": "ehr",
	"electronic health record": "ehr",
	"electronic medical record": "ehr",
	"laboratory value": "ehr",

	"ecg": "ecg",
	"ekg": "ecg",
	"electrocardiogram": "ecg",
	"holter": "ecg",
	"cardiac rhythm monitoring": "ecg",

	"ultrasound": "ultrasound",
	"echocardiography": "ultrasound",

	" ct ": "ct",
	"cardiac ct ": "ct",
	"coronary ct ": "ct",
	"coronary cta ": "ct",
	"ccta ": "ct",
	"computed tomography": "ct",

	"cardiac mr": "mr",
	" cmr ": "mr",
	" mri ": "mr",
	"magnetic resonance": "mr",

	"spect": "nuclear",
	" pet ": "nuclear",
	"positron emmission tomography": "nuclear",
	"pet-mri": "nuclear",
	"pet-ct": "nuclear",
	"nuclear cardiology": "nuclear",
	"nuclear card": "nuclear",
	"nuclear": "nuclear",

	"genetic": "genetic",
	"genomic": "genetic",
	"gene": "genetic",

	"pulse waveform":"waveform",
	"cardiotocographic":"waveform",
	"waveform":"waveform",
	"catheterization":"waveform",
	
}
import pandas as pd
keylist = list(set(data_modality_dict.keys()).union(set(dat	a_modality_dict.values())))
print(keylist)
finalwordlist = []
import csv
with open('./Refineddatamodalitylist.csv', mode='w', encoding='UTF-8', newline='') as file:
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
filesave = 'refineddatamodtopiclist.pickle'
with open(filesave, 'wb') as fp:
	pickle.dump(list(set(finalwordlist)) , fp)
print("--- %s seconds ---" % (time.time() - start_time))
print("done")