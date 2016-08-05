#!/usr/bin/python

import os, sys
#import shutil
import nltk
import random
file_paths1=[]
file_paths=[]
count = {}

DIR = r"C:\Users\DELL\AppData\Local\Programs\Python\Python35\mrc\pos"
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
#print(lpos[1])
      



DIR1 = r"C:\Users\DELL\AppData\Local\Programs\Python\Python35\mrc\neg"
for root,directories,files in os.walk(DIR1):
   for filename in files:
     filepath1=os.path.join(root,filename)
     file_paths1.append(filepath1)

for q in file_paths1:
   lnames=open(q,'r').read().split()
   lpos.append([lnames,'neg'])
   for w in lnames:
      all_words.append(w)
#print(lpos[1])
random.shuffle(lpos)
print(len(all_words))
#print(lpos)

word_features=list(all_words)[:2000]

#print(all_words)
def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

featuresets=[(document_features(d),c) for (d,c) in lpos]
train_set,test_set=featuresets[5:],featuresets[:5]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)                
