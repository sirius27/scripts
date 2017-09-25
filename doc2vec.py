#-*- coding:utf-8 -*-
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# random
from random import shuffle
# classifier
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import nltk
import time
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='xxxx')
#parser.add_argument('data_path',type=str,help='text data path')
args=parser.parse_args()
train_text_path = '../data/training_text'
test_text_path = '../data/testing_text'

class LabeledLineSentence():
    def __init__(self, sources):
        self.sources = sources
    
    def __iter__(self):
		for source, fname in self.sources.items():
			with utils.smart_open(fname) as fin:
				for item_no, line in enumerate(fin):
					yield LabeledSentence(utils.to_unicode(line).split(), ['%s_PARA_%d'%(source, item_no)])
    
    def to_array(self):
		self.sentences = []
		for source, fname in self.sources.items():
			with utils.smart_open(fname) as fin:
				for item_no, line in enumerate(fin):
					self.sentences.append(LabeledSentence( \
				      	utils.to_unicode(line).split(), ['%s_PARA_%d'%(source, item_no)]))
		return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources = {'TRAIN':train_text_path, 'TEST':test_text_path}
sentences = LabeledLineSentence(sources)
print sentences.to_array()[1]
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())
time_start = time.time()
model.train(sentences.sentences_perm(), total_examples=model.corpus_count,epochs=10)
time_end = time.time()
print 'task finished in %d seconds'%(time_end-time_start)
model.save('../models/model.d2v')
