ml_method_dict = {
	"neural network": "deeplearning",
	"deep learning": "deeplearning",
	"cnn": "deeplearning",

	"decision tree": "decisiontree",
	"random forest": "decisiontree",
	"tree-based": "decisiontree",
	
	"bayesian": "bayesian",
	"bayes": "bayesian",
	
	" svm ": "svm",
	"support vector machine": "svm",
	
	"clustering": "clustering",
	
	"boosting": "boosting",
	
	"logistic regression": "logisticregression",
	
	"statistical machine learning": "otherstatisticalML",
	"statistical ml": "otherstatisticalML",

	"artificial intelligence": "other",
	"machine learning": "other",
}


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



disease_dict = {
	"congenital heart disease": "congenitalheartdisease",

	"aortic stenosis": "valvedisease",
	"aortic valve": "valvedisease",
	"mitral regurgitation": "valvedisease",
	"mitral valve": "valvedisease",
	"valve": "valvedisease",

	"atherosclerosis": "atherosclerosis",
	"coronary artery disease": "atherosclerosis",
	" cad ": "atherosclerosis",
	"myocardial infarction": "atherosclerosis",
	"coronary artery lesion": "atherosclerosis",

	"stroke": "stroke",
	"cerebrovascular": "stroke",

	"atrial fibrillation": "arrhythmia",
	"atrial fib": "arrhythmia",
	" af ": "arrhythmia",
	"arrhythmia": "arrhythmia",

	"heart failure": "heartfailure",
	" chf": "heartfailure",
	" ahf ": "heartfailure",
	"heart transplant": "heartfailure",
	"cardiac transplant": "heartfailure",

}


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

"""
to see unique values in one of the dictionaries D obove:

dict_names = ("ml_method_dict", "ml_problem_dict", "disease_dict", "data_modality_dict")
dicts = ( ml_method_dict, ml_problem_dict, disease_dict, data_modality_dict )
for name, D in zip(dict_names, dicts):
	print("\n%s:" % name)
	for i in sorted(list(set( D.values() ))): print("\t%s" % i)
"""











import pandas as pd
import re
from string import rsplit


def replacer(i, replace_dict):
	i = str(i) # catches when i is a float or nan, although now i've done that in line ~625
	result = []
	for k, v in replace_dict.iteritems():
		if k in i: 
			result.append(v)
	if result == []: result = [""]
	result=list(set(result))
	return "_".join(result)



df = pd.read_csv('')

# df.count()

# [ ] change the Is Review column to an article type column. if review in Details column, the entry in Article type should be Review. If 'Erratum' in Details, put 'Erratum' in article type column. Look for other non-primary research article types, and annotate them in the Article Types column

df['Study Year'] = df['ShortDetails'].str.rsplit(' ',1).str[1]
df['Article Type'] = df['Details'].str.rsplit('.',2).str[1]

# wait there's already a Type column -- how did that come to pass?

# df["Is Review"] = ["True" if "Review" in cell else "False" for cell in df["Type"]]
# df["Is Erratum"] = ["True" if "Published Erratum" in cell else "False" for cell in df["Type"]]
# df["Is Editorial"] = ["True" if " Editorial" in cell else "False" for cell in df["Type"]]



df["Search"] = df["Title"].map(str) + df["Abstract"]
df['Search'] = df['Search'].replace('\n',' ', regex=True)
df['Search'] = df['Search'].str.replace('\ {2,}', ' ', regex=True)
df['Search'] = df['Search'].str.lower()
df['Is ML'] = df[df['Search'].str.contains('|'.join(ml_list))]

df = df.loc[df['Type'].str.contains("Journal Article", case=True) == True]


df = df.sort(['Is ML', 'Type', 'Study Year'], ascending=[0,1,2]) # new pandas
df = df.sort_values(['Is ML', 'Type', 'Study Year'], ascending=[0,1,2]) # old pandas



df['ML_problem'] = [replacer(i, ml_problem_dict) for i in df['Search']]

df['ML_method'] = [replacer(i, ml_method_dict) for i in df['Search']]

df['Disease'] = [replacer(i, disease_dict) for i in df['Search']]

df['Data_modality'] = [replacer(i, data_modality_dict) for i in df['Search']]





df = df.drop('Search', axis=1)

df.to_csv(r'', index=None, sep=',', mode='a') # put testfile path in ''

# df.count()
# pd.value_counts(df['Study Year'].values, sort = True)
# pd.value_counts(df['Is Review'].values, sort = True)

# the replacer dict method is really nice becuase it shows which things were found... 





# review was already removed here... 
>>> pd.value_counts(df['Type'].values, sort=True)
# Journal Article                                                                             50224
# Journal Article, Randomized Controlled Trial                                                 1212
# Journal Article, Multicenter Study                                                            791
# Journal Article, Multicenter Study, Randomized Controlled Trial                               784
# Clinical Trial, Journal Article                                                               650
# Clinical Trial, Journal Article, Randomized Controlled Trial                                  554
# Comment, Editorial                                                                            547
# Letter                                                                                        364
# Comment, Letter                                                                               341
# Journal Article, Meta-Analysis                                                                333
# Editorial                                                                                     318
# Comment, Journal Article                                                                      267
# Clinical Trial, Journal Article, Multicenter Study, Randomized Controlled Trial               198
# Clinical Trial, Journal Article, Multicenter Study                                            101
# News                                                                                           98
# Journal Article, Practice Guideline                                                            78
# Journal Article, Retracted Publication                                                         55
# Guideline, Journal Article, Practice Guideline                                                 25
# Historical Article, Journal Article                                                            21
# Journal Article, Review                                                                        20
# Consensus Development Conference, Journal Article                                              19
# Interview                                                                                      14
# Letter, Randomized Controlled Trial                                                            13
# Consensus Development Conference, Journal Article, Practice Guideline                          12
# Comment, News                                                                                  11
# Review                                                                                         10
# Clinical Trial, Letter                                                                          9
# Journal Article, Published Erratum                                                              7
# Letter, Multicenter Study, Randomized Controlled Trial                                          7
# Guideline, Journal Article                                                                      5
# Corrected and Republished Article, Journal Article                                              5
# Published Erratum                                                                               5
# Journal Article, Patient Education Handout                                                      5
# Comment, Journal Article, Patient Education Handout                                             4
# Biography, Historical Article, Journal Article                                                  4
# Patient Education Handout                                                                       3
# Letter, Meta-Analysis                                                                           3
# Journal Article, Meta-Analysis, Randomized Controlled Trial                                     3
# Journal Article, Randomized Controlled Trial, Retracted Publication                             3
# Clinical Trial, Letter, Randomized Controlled Trial                                             3
# Journal Article, Meta-Analysis, Multicenter Study                                               3
# Clinical Trial, Journal Article, Randomized Controlled Trial, Retracted Publication             2
# Comment, Journal Article, Randomized Controlled Trial                                           2
# Letter, Multicenter Study                                                                       2
# Clinical Trial, Comment, Journal Article, Randomized Controlled Trial                           2
# Classical Article, Journal Article                                                              1
# Journal Article, Meta-Analysis, Multicenter Study, Randomized Controlled Trial                  1
# Comment, Letter, Retracted Publication                                                          1
# Journal Article, Multicenter Study, Practice Guideline                                          1
# Historical Article, Letter                                                                      1
# Classical Article, Historical Article, Journal Article                                          1
# Clinical Trial, Comment, Letter, Randomized Controlled Trial                                    1
# Classical Article, Journal Article, Multicenter Study                                           1
# Biography, Historical Article, News                                                             1
# Comment, Letter, Randomized Controlled Trial                                                    1
# Clinical Trial, Comment, Journal Article, Multicenter Study, Randomized Controlled Trial        1
# Guideline, Journal Article, Multicenter Study, Practice Guideline                               1
# News, Practice Guideline                                                                        1
# Clinical Trial, Journal Article, Meta-Analysis                                                  1
# Clinical Trial, Journal Article, Retracted Publication                                          1
# Clinical Trial, Letter, Multicenter Study                                                       1
# Journal Article, Multicenter Study, Randomized Controlled Trial, Retracted Publication          1
# Editorial, Meta-Analysis                                                                        1
# Multicenter Study, News, Randomized Controlled Trial                                            1
# Editorial, Historical Article                                                                   1
# Overall                                                                                         1
# Newspaper Article                                                                               1
# Comment, Journal Article, Multicenter Study, Randomized Controlled Trial                        1
# Biography, Historical Article, Interview                                                        1
# Clinical Trial, Editorial                                                                       1
# Clinical Trial, Comment, Editorial, Randomized Controlled Trial                                 1
# Clinical Trial, News, Randomized Controlled Trial                                               1


ml_list = ["kernel", "fourier transform", "cascade of classifiers", "machine learning", "deep learning", "neural network", "network", "support vector machine", "random forest", "boosting", "algorithm", "supervised", "unsupervised", "classification", "regression", "logistic regression", "generative model", "discriminative model", "statistical ML", "learning algorithm", "bagging", "bayesian network", "bayesian algorithm", "naive bayes", "adaboost", "xgboost", "reinforcement learning", "artificial intelligence", "backpropagation", "bag of words", "tfidf", "tf-idf", "batch normalization", "object tracking", "semantic segmentation", "centroid-based clustering", "classification model", "classification algorithm", "segmentation model", "segmentation algorithm", "clustering", "confusion matrix", "convolutional", "cnn", "convolutional layer", "cross-validation", "cross-entropy", "data augmentation", "decision tree", "dimensionality reduction", "discriminative model", "dropout regularization", "embeddings", "feature engineering", "feature selection", "federated learning", "feed forward neural network", "one shot learning", "one-shot learning", "few-shot learning", "few shot learning", "softmax", "fully connected layer", "generative adversarial network", "adversarial learning", "ground truth", "hidden layer", "hierarchical clustering", "k-means", "learning rate", "learning function", "nearest neighbors", "LSTM", "long short-term memory", "long short term memory", "markov", "stochastic gradient descent", "multi-class logistic regression", "natural language processing", " nlp ", "n-gram", "tensorflow", "pytorch", "keras", "batch normalization", "numpy", "one-hot vector", "one-hot encoding", "overfitting", "perceptron", "random forest", " relu ", "rectified linear unit", "recurrent neural network", " rnn ", "reinforcement learning", "scikit-learn", "semi-supervised learning", " gpu ", "graphics processing unit", "computer vision", "data science", "data engineering", "feature reduction", "principal components analysis", "principal component analysis", "hidden markov model", "hierarchical clustering", " knn ", "monte carlo", "stochastic gradient descent", "transfer learning", "style transfer"]


# phrases that are too non-specific for ML:
# 'bayesian' alone
# 'logistic regression' alone
# 'automated software' alone





flags for type non-primary research
the pubmed flags
overview
review
summary
if the journal has the word review in it. like nature reviews
if the article has no Abstract
