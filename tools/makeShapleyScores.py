"""
python makeShapleyScores.py [vocab] [corpus1] [corpus2] r
subsets a [vocab] file by finding the words most associated with
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
uses ngrams
"""
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
            
          
            self.vectorizerSource  = CountVectorizer(tokenizer=tokenize)  
            self.vectorizerTarget  = CountVectorizer(tokenizer=tokenize) 
            self.sourceMatrix      = self.vectorizerSource.fit_transform(sourceCorpus+targetCorpus[:50])
            self.sourceVocab       = self.vectorizerSource.vocabulary_
            self.targetMatrix      = self.vectorizerTarget.fit_transform(targetCorpus+sourceCorpus[:50])
            self.targetVocab       = self.vectorizerTarget.vocabulary_
            self.featureSourceName = self.vectorizerSource.get_feature_names()
            self.featureTargetName = self.vectorizerTarget.get_feature_names()

            
            self.labelsSource      = np.concatenate((np.zeros(len(sourceCorpus), dtype=np.int8), np.ones(50, dtype=np.int8)))
            self.labelsTarget      = np.concatenate((np.ones(len(targetCorpus),  dtype=np.int8), np.zeros(50, dtype=np.int8)))

            
    def trainNgramClassifier(self):
        # fit the training dataset on the classifier
        classifierSource = xgboost.XGBClassifier(n_jobs=-1)
        classifierSource.fit(self.sourceMatrix, self.labelsSource)
        classifierTarget = xgboost.XGBClassifier(n_jobs=-1)
        classifierTarget.fit(self.targetMatrix, self.labelsTarget)

        self.explainerSource  = shap.TreeExplainer(classifierSource)
        self.explainerTarget  = shap.TreeExplainer(classifierTarget)
        
    def getTransformClassifier(self):
        classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)
        return classifier
     
    def ngramShapley(self, data, dataIndex, lmbda=0.5):
        sourceValues = self.explainerSource(self.sourceMatrix[dataIndex]).values
        postValues = self.explainerTarget(self.targetMatrix[dataIndex]).values
        #https://towardsdatascience.com/hey-model-why-do-you-say-this-is-spam-7c945cc531f
        ngramValues  = {}
        #To-do: validar
        for gram in tokenize(data):
             
            if gram not in self.featureSourceName:
                i = 0.0
            else:
                i = sourceValues[0, self.featureSourceName.index(gram[0])]
            
            if gram not in self.featureTargetName:
                j = 0.0
            else:
                j = postValues[0, self.featureTargetName.index(gram[0])] 
            ngramValues[gram] = ((i + lmbda) / (j + lmbda), (j + lmbda) / (i + lmbda))
            
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
            ngramValues[gram] = ((i + lmbda) / (j + lmbda), (j + lmbda) / (i + lmbda))
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
sc.trainNgramClassifier()
#para ambos fazer o for ngram nas respectivas Funcoes
    #para o transformer fazer o sum 
    #para o outro não
    #osar shapley puro e não normalizado
    
print("marker", "negative_score", "positive_score")
def calculate_attribute_markers(corpus):
    for i, sentence in enumerate(tqdm(corpus)):
        salience = sc.ngramShapley(sentence, i)
        print(salience)
        input()

        #if max(negativeSalience, positiveSalience) > r:
            #print(gram, negativeSalience, positiveSalience)
            #print(gram)


calculate_attribute_markers(corpus1)
calculate_attribute_markers(corpus2)