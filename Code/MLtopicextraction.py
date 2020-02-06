import gensim
import time
start_time = time.time()
model = gensim.models.KeyedVectors.load_word2vec_format('./BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)
print("--- %s seconds ---" % (time.time() - start_time))

import pandas as pd
keypd = pd.read_csv('./Cardiolgykeywords.csv', header = None)
keylist = list(keypd.values.flatten())
keylist = ml_list = ["kernel", "fourier transform", "cascade of classifiers", 
"machine learning", "deep learning", "neural network", "network", "support vector machine", 
"random forest", "boosting", "algorithm", "supervised", "unsupervised", "classification", 
"regression", "logistic regression", "generative model", "discriminative model", 
"statistical ML", "learning algorithm", "bagging", "bayesian network", 
"bayesian algorithm", "naive bayes", "adaboost", "xgboost", "reinforcement learning", 
"artificial intelligence", "backpropagation", "bag of words", "tfidf", "tf-idf", 
"batch normalization", "object tracking", "semantic segmentation", "centroid-based clustering",
 "classification model", "classification algorithm", "segmentation model", 
 "segmentation algorithm", "clustering", "confusion matrix", "convolutional", "cnn",
  "convolutional layer", "cross-validation", "cross-entropy", "data augmentation",
   "decision tree", "dimensionality reduction", "discriminative model", "dropout regularization",
   "embeddings", "feature engineering", "feature selection", "federated learning", 
   "feed forward neural network", "one shot learning", "one-shot learning", 
   "few-shot learning", "few shot learning", "softmax", "fully connected layer", 
   "generative adversarial network", "adversarial learning", "ground truth", "hidden layer", 
   "hierarchical clustering", "k-means", "learning rate", "learning function", 
   "nearest neighbors", "LSTM", "long short-term memory", "long short term memory", "markov",
    "stochastic gradient descent", "multi-class logistic regression", 
    "natural language processing", " nlp ", "n-gram", "tensorflow", "pytorch", "keras", 
    "batch normalization", "numpy", "one-hot vector", "one-hot encoding", "overfitting", 
    "perceptron", "random forest", " relu ", "rectified linear unit", "recurrent neural network",
     " rnn ", "reinforcement learning", "scikit-learn", "semi-supervised learning", " gpu ",
      "graphics processing unit", "computer vision", "data science", "data engineering", 
      "feature reduction", "principal components analysis", "principal component analysis", 
      "hidden markov model", "hierarchical clustering", " knn ", "monte carlo", 
      "stochastic gradient descent", "transfer learning", "style transfer"]
print(keylist)
finalwordlist = []
import csv
with open('./RefinedMLtopicdist.csv', mode='w', encoding='UTF-8', newline='') as file:
	ewrite = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	ewrite.writerow(['Phrase', 'MLVal', 'NNVal'])
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
					cval = model.similarity('machine-learning', phrase)
				except:
					cval = 0
				if cval != 0:
					finalwordlist.append(phrase)
					hval = model.similarity('neural-network', phrase)
					ewrite.writerow([phrase, cval, hval])
print("--- %s seconds ---" % (time.time() - start_time))

import pickle
filesave = 'refinedMLtopiclist.pickle'
with open(filesave, 'wb') as fp:
	pickle.dump(list(set(finalwordlist)) , fp)
print("--- %s seconds ---" % (time.time() - start_time))
print("done")
print(finalwordlist)