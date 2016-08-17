#!/usr/bin/python

import os, sys
import nltk
import random
file_paths1=[]
file_paths=[]
count = {}

DIR = "/home/rohith/nltk_data/corpora/movie_reviews/pos"
for root,directories,files in os.walk(DIR):
   for filename in files:
     filepath=os.path.join(root,filename)
     file_paths.append(filepath)

all_words=[]
lnames=[]
lpos=[[[],'pos']]
for p in file_paths:
   lnames=open(p,'r').read().split()
   lpos.append([lnames,'pos'])
   for w in lnames:
      all_words.append(w)
      

DIR1 = "/home/rohith/nltk_data/corpora/movie_reviews/neg"
for root,directories,files in os.walk(DIR1):
   for filename in files:
     filepath1=os.path.join(root,filename)
     file_paths1.append(filepath1)

for q in file_paths1:
   lnames=open(q,'r').read().split()
   lpos.append([lnames,'neg'])
   for w in lnames:
      all_words.append(w)

random.shuffle(lpos)
print(len(all_words))

word_features=list(all_words)[:2000]

def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

featuresets=[(document_features(d),c) for (d,c) in lpos]
train_set,test_set=featuresets[5:],featuresets[:5]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Accuracy in %:")
print (nltk.classify.accuracy(classifier, test_set)*100)
classifier.show_most_informative_features(5)                
