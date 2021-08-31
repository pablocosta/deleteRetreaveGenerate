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
from transformers import BertTokenizer, BertModel
import torch
from keybert import KeyBERT



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
            self.classifier, self.tokenizer = self.getTransformClassifier()
            self.sourceData         = self.transformerRepresentation(sourceCorpus)
            self.targetData         = self.transformerRepresentation(targetCorpus)
            self.valuesCountsTarget = self.sumValues(self.targetData)
            self.valuesCountsSource = self.sumValues(self.sourceData)
            
        elif shapleyForm == "ngram":
            
          
            self.vectorizerSource  = CountVectorizer(tokenizer=tokenize)  
            self.vectorizerTarget  = CountVectorizer(tokenizer=tokenize) 
            self.sourceMatrix      = self.vectorizerSource.fit_transform(sourceCorpus+targetCorpus[:50])
            self.sourceVocab       = self.vectorizerSource.vocabulary_
            self.targetMatrix      = self.vectorizerTarget.fit_transform(targetCorpus+sourceCorpus[:50])
            self.targetVocab       = self.vectorizerTarget.vocabulary_
            featureSourceName = self.vectorizerSource.get_feature_names()
            featureTargetName = self.vectorizerTarget.get_feature_names()

            
            self.labelsSource      = np.concatenate((np.zeros(len(sourceCorpus), dtype=np.int8), np.ones(50, dtype=np.int8)))
            self.labelsTarget      = np.concatenate((np.ones(len(targetCorpus),  dtype=np.int8), np.zeros(50, dtype=np.int8)))
            self.dsctNameSource = {}
            self.dsctNameTarget = {}
            for i, k in enumerate(featureSourceName):
                self.dsctNameSource[k]= i

            for i, k in enumerate(featureTargetName):
                self.dsctNameTarget[k]= i
        elif shapleyForm == "keybert":
            self.keyWordBert = KeyBERT()
            self.dictKeyBertSource = self.extractNgramKeyBert(sourceCorpus)
            self.dictKeyBertTarget = self.extractNgramKeyBert(targetCorpus)
            
            
            
    def trainNgramClassifier(self):
        # fit the training dataset on the classifier
        classifierSource = xgboost.XGBClassifier(n_jobs=-1, max_depth=2, predictor="gpu_predictor")
        classifierSource.fit(self.sourceMatrix, self.labelsSource)
        classifierTarget = xgboost.XGBClassifier(n_jobs=-1, max_depth=2, predictor="gpu_predictor")
        classifierTarget.fit(self.targetMatrix, self.labelsTarget)

        self.explainerSource  = shap.TreeExplainer(classifierSource, feature_perturbation="tree_path_dependent")
        self.explainerTarget  = shap.TreeExplainer(classifierTarget, feature_perturbation="tree_path_dependent")
        
    def getTransformClassifier(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        classifier = BertModel.from_pretrained('bert-base-uncased')
        return classifier, tokenizer
     
    def ngramShapley(self, data, dataIndex, lmbda=0.5):
        
        sourceValues = self.explainerSource(self.vectorizerSource.transform([data])).values
        postValues = self.explainerTarget(self.vectorizerTarget.transform([data])).values
        #https://towardsdatascience.com/hey-model-why-do-you-say-this-is-spam-7c945cc531f
        ngramValues  = {}
        #To-do: validar
        for gram in tokenize(data):
            
            try:
                i = sourceValues[0, self.dsctNameSource[gram]]
            except:
                i = 0.0
            try:
                j = postValues[0, self.dsctNameTarget[gram]]
            except:
                j = 0.0
            ngramValues[gram] = ((i + lmbda) / (j + lmbda), (j + lmbda) / (i + lmbda))
        return ngramValues        
    
    def sumValues(self, valuesData):
        
        for k in valuesData.keys():
            valuesData[k] = np.sum(valuesData[k])
        return valuesData
    
    def extractNgramKeyBert(self, data):
        ngramValues = {}
        for text in tqdm(data):
            bert = self.keyWordBert.extract_keywords(text, keyphrase_ngram_range=(1, 5))
            for pair in bert:
                if pair[0] in ngramValues:
                    ngramValues[pair[0]].append(pair[1])
                else:
                    ngramValues[pair[0]] = [pair[1]]
        return ngramValues
    def ngramTransformer(self, data, lmbda=0.3):
        ngramValues  = {}
        #To-do: validar
        for gram in tokenize(data):
            
            if gram in self.valuesCountsSource:
                i = self.valuesCountsSource[gram]
            else:
                i = 0.0
                
            if gram in self.valuesCountsTarget:
                j = self.valuesCountsTarget[gram]
            else:
                j = 0.0
                
            ngramValues[gram] = ((i + lmbda) / (j + lmbda), (j + lmbda) / (i + lmbda))
        return ngramValues 
    def ngramKeyBert(self, data, lmbda=0.3):
        ngramValues  = {}
        #To-do: validar
        for gram in tokenize(data):
            
            if gram in self.dictKeyBertSource:
                i = np.mean(self.dictKeyBertSource[gram])
            else:
                i = 0.0
                
            if gram in self.dictKeyBertTarget:
                j = np.mean(self.dictKeyBertTarget[gram])
            else:
                j = 0.0
                
            ngramValues[gram] = ((i + lmbda) / (j + lmbda), (j + lmbda) / (i + lmbda))
        return ngramValues 
    
    
    
    def transformerRepresentation(self, data):
        ngramValues = {}
        for text in tqdm(data):
            inputs  = self.tokenizer(text, return_tensors="pt")
            outputs = self.classifier(**inputs)
            last_hidden_states = outputs.last_hidden_state
            splittedText = text.split(" ")
  
            for gram in tokenize(text):

                media = np.mean(last_hidden_states[0, splittedText.index(gram.split(" ")[0]):splittedText.index(gram.split(" ")[0])+len(gram.split(" "))].detach().numpy())
                if gram in ngramValues:
                    ngramValues[gram.lower()].append(media)
                else:
                    ngramValues[gram.lower()] = [media] 
                    
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


sc = shapleyCalculator(corpus1, corpus2, tokenize, shapleyForm="keybert")

#sc.trainNgramClassifier()
#para ambos fazer o for ngram nas respectivas Funcoes
    #para o transformer fazer o sum 
    #para o outro não
    #osar shapley puro e não normalizado
    
print("marker", "negative_score", "positive_score")
def calculate_attribute_markers(corpus):
    for i, sentence in enumerate(tqdm(corpus)):
        salience = sc.ngramKeyBert(sentence)
        for k in salience.keys():
            negativeSalience, positiveSalience = salience[k]
            if max(negativeSalience, positiveSalience) > r:
                print(k, negativeSalience, positiveSalience)
            


calculate_attribute_markers(corpus1)
calculate_attribute_markers(corpus2)