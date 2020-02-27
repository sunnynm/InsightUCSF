# Code File Summary

## Cardtopicextraction.py

Inputs a small list of reference keywords/key phrases to use for better determining what extracted keywords are most relevant and then checks which reference keywords are in the word2vec model and outputs the list in a pickle as well as a csv showing distance to the interested base topic word.

## Fullbio2vecapplication.py

Takes in the 2 verified list of reference keywords (refined is the list outputted by cardtopicextracton while manuallyrefined is the list after removing key subphrases that would not be useful such as 'tree' from 'binary tree') and the csv containing the title and abstracts of the documents to. Extracts keywords and then determines the relevance to reference keywords. Outputs a csv containing key phrase extracted, the associated document, the cosine similarity to the base interested word, and then the word in the keyword list it was most similar to along with the cosine similarity to that word.

## Wordcloudextraction.py

Creates word cloud based on the keyword distance csv outputted from Fullbio2vecapplication. In our case we manually looked at a subset of the results and then calculated a threshold we thought would give the least amount of false positives and cross-applied that threshold to the full dataset

## KeywordPlot.py

Clustering of embeddings of the keywords extracted to see if the related words are properly being clustered together. Keywords were selected by threshold as described above. 

## DocumentClusterPlot.py

Uses all extracted keywords/key phrases embeddings to cluster the documents. Keywords were selected by threshold as described above.
