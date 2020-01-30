import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import csv

import time
start_time = time.time()

subdf = pd.read_csv('AllAbstracts.csv')
documentlist = subdf['Abstract'].values
documentlist = [re.sub('[0-9]+', 'XXX', document) for document in documentlist]
documenttitles = subdf['Title'].values
for gram in range(2,5):
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(gram,gram))
    X = vectorizer.fit_transform(documentlist)
    words = vectorizer.get_feature_names()
    base = './TFIDF{}keywordsFULL.csv'.format(str(gram)+"gram")
    with open(base, mode='w', encoding='UTF-8', newline='') as file:
        ewrite = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ewrite.writerow(['Document', 'Word', 'TFIDFVal'])
        for doc in range(len(documentlist)) :
            wordlist = X[doc,:].nonzero()[1]
            vallist = [X[doc, x] for x in wordlist]
            topwords = sorted(range(len(vallist)), key=lambda i: vallist[i])[-10:]
            subwords = [words[i] for i in wordlist[topwords]][::-1]
            subval = [vallist[i] for i in topwords][::-1]
            for i in range(len(subval)):
                ewrite.writerow([documenttitles[doc], subwords[i], subval[i]])
    print("--- %s seconds ---" % (time.time() - start_time))
print("done")
        