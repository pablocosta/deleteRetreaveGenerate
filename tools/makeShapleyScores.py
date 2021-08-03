import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition, ensemble
import transformers
from transformers.utils.dummy_pt_objects import FunnelForSequenceClassification, FunnelForTokenClassification, Trainer
import xgboost
import shap 
import sys 


def tokenize(text):
    text = text.split()
    grams = []
    for i in range(1, 5):
        i_grams = [
            " ".join(gram)
            for gram in ngrams(text, i)
        ]
        grams.extend(i_grams)
    return grams

class shapleyCalculator(object):
    
    def __init__(self, sourceCorpus, targetCorpus, tokenize, shapleyForm):
        
        if shapleyForm == "transformer":
            self.classifier = self.getTransformClassifier()
            self.explainer = shap.Explainer(self.classifier)
            
        else:

            
            self.vectorizer    = CountVectorizer(tokenizer=tokenize)  
            sourceMatrix       = self.vectorizer.fit_transform(sourceCorpus)
            self.sourceVocab   = self.vectorizer.vocabulary_
            targetMatrix  = self.vectorizer.fit_transform(targetCorpus)
            self.targetVocab   = self.vectorizer.vocabulary_
            _ = self.vectorizer.fit_transform(sourceCorpus + targetCorpus)
            
            
            labels             = np.concatenate((np.zeros(len(sourceCorpus)), np.ones(len(targetCorpus))), axis=0)
            labelsSource = np.zeros(len(sourceCorpus))
            labelsTarget = np.ones(len(targetCorpus))
            
            #train classifiers
            self.classifierSource, self.classifierTarget = self.trainNgramClassifier(xgboost.XGBClassifier(), xgboost.XGBClassifier(), sourceMatrix, targetMatrix, labelsSource, labelsTarget, labels)
            self.explainerSource     = shap.TreeExplainer(self.classifierSource)
            self.explainerTarget     = shap.TreeExplainer(self.classifierTarget)
            
    def trainNgramClassifier(self, classifierSource, classifierTarget, featureVectorSource, featureVectorTarget, labelsSource, labelsTarget):
        # fit the training dataset on the classifier
        
        classifierSource.fit(featureVectorSource, labelsSource)
        classifierTarget.fit(featureVectorTarget, labelsTarget)
        
        return classifierSource, classifierTarget
    def getTransformClassifier(self):
        classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)
        return classifier
     
    def ngramShapley(self, data, lmbda=0.5):
        sourceValues = self.explainerSource.shap_values(self.vectorizer.transform([data]))
        postValues   = self.classifierTarget.shap_values(self.vectorizer.transform([data]))
        ngramValues  = {}
        #To-do: validar
        for gram in tokenize(data):
            i = sourceValues[self.vectorizer.vocabulary_[gram]]
            j = postValues[self.vectorizer.vocabulary_[gram]]
            ngramValues[gram] = [(i + lmbda) / (j + lmbda), (j + lmbda) / (i + lmbda)]
        
        return ngramValues        
        
    def transformerShapley(self, data, lmbda=0.5):
        
        # explain the predictions of the pipeline on the first two samples
        shapValues  = self.explainer(data)
        
        sourceValues = shapValues[:,"POSITIVE"]
        postValues   = shapValues[:,"NEGATIVE"]
        ngramValues  = {}
        for gram in tokenize(data):
            
            i = np.sum([sourceValues[data.split(" ").index(i)] for i in gram]), np.sum([sourceValues[data.split(" ").index(i)] for i in gram])
            j = np.sum([sourceValues[data.split(" ").index(i)] for i in gram]), np.sum([postValues[data.split(" ").index(i)] for i in gram])
            ngramValues[gram] = [(i + lmbda) / (j + lmbda), (j + lmbda) / (i + lmbda)]
                #ngramValues[gram] = [sourceValues[self.vectorizer.vocabulary_[gram]], postValues[self.vectorizer.vocabulary_[gram]]]
        
                    
        return ngramValues                
    
    
    
# create a set of all words in the vocab
vocab = set([w.strip() for i, w in enumerate(open(sys.argv[1]))])

corpus1_sentences = [
    l.strip().split()
    for l in open(sys.argv[2])
]

corpus2_sentences = [
    l.strip().split()
    for l in open(sys.argv[3])
]



# the salience ratio
r = float(sys.argv[4])

def unk_corpus(sentences):
    corpus = []
    for line in sentences:
        # unk the sentence according to the vocab
        line = [
            w if w in vocab else '<unk>'
            for w in line
        ]
        corpus.append(' '.join(line))
    return corpus


corpus1 = unk_corpus(corpus1_sentences)
corpus2 = unk_corpus(corpus2_sentences)


sc = shapleyCalculator(corpus1, corpus2, tokenize, "ngram")
#para ambos fazer o for ngram nas respectivas Funcoes
    #para o transformer fazer o sum 
    #para o outro não
    #osar shapley puro e não normalizado
    
print("marker", "negative_score", "positive_score")
def calculate_attribute_markers(corpus):
    for sentence in tqdm(corpus):
        salience = sc.ngramShapley(sentence)
        print(salience)
        input()

        #if max(negativeSalience, positiveSalience) > r:
            #print(gram, negativeSalience, positiveSalience)
            #print(gram)


calculate_attribute_markers(corpus1)
calculate_attribute_markers(corpus2)