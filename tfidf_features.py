#-*- coding:utf-8 -*-
import logging
logging.basicConfig()
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def get_cv(str_lst):
	vectorizer = CountVectorizer()
	counts = vectorizer.fit_transform(str_lst)
	return counts

def get_tfidf(str_lst):
	counts = get_cv(str_lst)
	tfidf = TfidfTransformer(use_idf=False).fit_transform(counts)
	return tfidf

def get_tf_cv(text_path):
	with open(text_path) as f:
		lines = f.readlines()
	cv = np.array(get_cv(lines).todense())
	tfidf = np.array(get_tfidf(lines).todense())
	return tfidf, cv
	

if __name__ == '__main__':
	train_text_path = '../temp_data/train_reduced.txt'
	test_text_path = '../temp_data/test_reduced.txt'
	print '计算训练集tfidf, cv'
	train_tfidf, train_cv = get_tf_cv(train_text_path)
	print '计算测试集tfidf, cv'
	test_tfidf, test_cv = get_tf_cv(test_text_path)
	np.save('../temp_data/count_vector/train_count_vector.npy', train_cv)
	np.save('../temp_data/tfidf/train_tf_idf_vector.npy', train_tfidf)	
	np.save('../temp_data/count_vector/test_count_vector.npy', test_cv)
	np.save('../temp_data/tfidf/test_tf_idf_vector.npy', test_tfidf)
