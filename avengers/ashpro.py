import nltk
import random

from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from nltk.classify import ClassifierI #inherhit classify class
#this program done in anaconda if using python use statistics instead of scipy.stats
from scipy.stats import mode #choose votes

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        
    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        



documents = [(list(movie_reviews.words(fileid)), category) 
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
                  
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower()) #lower case all word
    
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:8500]


def find_features(document):
    words = set(document)
    features = {}   
    for w in word_features:
       features[w] = (w in words)
        
    return features
    
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev,category) in documents]
#for positive values
training_set = featuresets[:1900]
testing_set = featuresets[1900:]
#for negative values
training_set = featuresets[50:]
testing_set = featuresets[:50]

classifier = nltk.NaiveBayesClassifier.train(training_set)


print(" original naives bayes algo accuracy percent:", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("  mnb classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)

#GaussianNB, BernoulliNB



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("  bernoulli classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

# LogisticRegression, SGDClassifier
# SVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("  logistic regression classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("  SGDClassifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("  LinearSVC accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("  NuSVC classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


#doing the voting of all classifier to get accuracy more 

voted_classifier = VoteClassifier(
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  
                                  LinearSVC_classifier,
                                  NuSVC_classifier)
print("  voted classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier,testing_set))*100)































