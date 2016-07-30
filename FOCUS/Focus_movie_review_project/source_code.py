import os
import random
import nltk
from os import listdir
from os.path import isfile,join
path = os.path.dirname(__file__)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
posfiles=os.path.join(path,'pos\\')
negfiles=os.path.join(path,'neg\\')


list_pos_pathname=[]
list_neg_pathname=[]
full_words=[]
open_pos=[]
lall=[[[],'pos']]



from os import walk

for root, directories, files in os.walk(posfiles):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            pathname= os.path.join(root, filename)
            list_pos_pathname.append(pathname)
            
            
for p in list_pos_pathname:
        
        open_pos=open(p,'r').read().split()
        lall.append([open_pos,'pos'])
        for words in open_pos:
                full_words.append(words)
         
         
for root, directories, files in os.walk(negfiles):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            pathname1= os.path.join(root, filename)
            list_neg_pathname.append(pathname1)
            
            
for n in list_neg_pathname:
        
        open_neg=open(n,'r').read().split()
        lall.append([open_neg,'neg'])
        for words in open_neg:
                full_words.append(words)


print("total no of tagged words is",len(full_words))
random.shuffle(full_words)
word_features=list(full_words)[:1000]

#print(all_words)
def document_features(docs): 
    document_words = set(docs) 
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

featuresets=[(document_features(d),c) for (d,c) in lall]
train_set,test_set=featuresets[100:],featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5) 
                


              



