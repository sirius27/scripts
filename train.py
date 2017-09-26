#-*- coding:utf-8 -*-
import argparse
import os
import tqdm

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch as t

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec
from MultiCNNTextDeep import MultiCNN
from torch import optim, nn



parser = argparse.ArgumentParser()
parser.add_argument('model_path',type=str,help='model file path')
parser.add_argument('cv_path', type=str, help='path to folder with vector count features')
parser.add_argument('tf_path', type=str,help='path to folder with tf-idf features')
args = parser.parse_args()

train_variants_path='../data/training_variants'
test_variants_path='../data/testing_variants'

def get_vg(path):
	df = pd.read_csv(path)
	print df.head()
	variants = df['Variation'].values
	genes = df['Gene'].values
	return variants, genes

def organize(model, train_len, test_len):
	docvecs = model.docvecs
	train_keys = ['TRAIN_PARA_%d'%i for i in range(train_len)]
	test_keys = ['TEST_PARA_%d'%i for i in range(test_len)]
	
	train_vecs = docvecs[train_keys]
	test_vecs = docvecs[test_keys]
	return train_vecs, test_vecs

def get_d2v_data(model_path):
	model=Doc2Vec.load(model_path)
	train_variants, train_genes = get_vg(train_variants_path)
	test_variants, test_genes = get_vg(test_variants_path)
	train_data, test_data = organize(model, len(train_variants), len(test_variants))
	print train_data.shape, len(train_variants)
	print test_data.shape, len(test_variants)
	#训练集标签
	train_variants = pd.read_csv(train_variants_path)
	#lightgbm需要[0,n)类标签，而这里给出的是[1,n+1)
	train_labels = train_variants['Class'].values-1
	return train_data, test_data, train_labels

def get_tfidf_data(tf_path):
	#tf path 下应该有tf train和tf test, 为csv或者npy
	train_tf = np.load(os.path.join(tf_path, 'train_features.npy'))
	test_tf = np.load(os.path.join(tf_path, 'test_features.npy'))
	return train_tf, test_tf

def get_count_vector(cv_path):
	#tf path 下应该有tf train和tf test, 为csv或者npy
	train_cv = np.load(os.path.join(cv_path, 'train_features.npy'))
	test_cv = np.load(os.path.join(cv_path, 'test_features.npy'))
	return train_cv, test_cv

def train(train_data, train_labels):
    #首先用lgb训练
    print 'training gbm'
    lgb_train = lgb.Dataset(train_data,train_labels)
	#lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
	# specify your configurations as a dict
    params = {'task': 'train','boosting_type': 'gbdt','application': 'multiclass','metric': {'multi_logloss'},\
		  'num_leaves': 31,'learning_rate': 0.05,'feature_fraction': 0.9,'bagging_fraction': 0.8,\
		  'bagging_freq': 5,'verbose': 0, 'num_class':9}
    print 'Start training...'
	#训练.暂时没有验证集
    gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=None)
	#其次用logistic regression
    print 'training linear regression'
    clf_lr = LogisticRegression()
    clf_lr.fit(train_data, train_labels)
	#然后用svm
    print 'training svm'
    clf_svm = SVC(probability = True)
    clf_svm.fit(train_data, train_labels)
    #最后用cnn
	print 'training cnn'
	cnn = MultiCNNDeepText()
    t = tqdm.trange(0, args.nb_epoch, desc='cnn')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr = args.learning_rate)
    best_loss = -1
    for i in t:
        running_loss = 0
        for batch_start in range(0, train_X.shape[0], args.batch_size):
            loss.zero_grad()
            batch_end = batch_end + args.batch_size

            input_X = Variable(train_X[batch_start:batch_end,:].from_numpy())
            real_y = Variable(train_y[batch_start:batch_end].from_numpy())

            output = cnn(input_X)
            batch_loss = criterion(output, real_y)
            running_loss += batch_loss.data.numpy()
            loss.backward()
            optimizer.step()
        if best_loss < 0:
            best_loss = running_loss
        elif running_loss < best_loss:
            best_loss = running_loss
	        state_dict = {'epoch':epoch+1,'state_dict':model.state_dict(),'best_mse':best_prec1,'optimizer':optimizer.state_dict()}
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            t.save(state_dict, '../models/'+current_time+'_checkpoint.pth')
        t.set_description('mse = %.6f' % running_loss)

    return gbm, clf_lr, clf_svm

def one_hot_encode(y, nb_classes):
	y_pred = [np.argmax(i) for i in y]
	one_hot = np.zeros((len(y_pred), nb_classes))
	print one_hot.shape
	print nb_classes
	for i,c in enumerate(y_pred):
		one_hot[i,c]=1
	return one_hot

def output(y):
	nb_samples = y.shape[0]
	nb_classes = y.shape[1]
	classes = ['class%d'%(i+1)for i in range(nb_classes)]
	ids = np.arange(nb_samples).reshape(nb_samples,1)
	output_df = pd.DataFrame(np.concatenate([ids, y], axis = 1), columns = ['ID'] + classes)
	output_df = output_df.set_index(['ID'])
	output_df.to_csv('../results/result.csv')

def feature_concat(*args):
	return reduce(lambda x,y: np.concatenate([x,y], axis = 1), args)

def ensemble_prediction(*args):
	new_prediction = np.zeros_like(args[0])
	for arg in args:
		#normalize
		print arg.shape
		print arg.sum()
		new_prediction += arg/arg.sum()
	return new_prediction/len(args)

def main():
	#读取独立features
	d2v_train_features, d2v_test_features, train_labels = get_d2v_data(args.model_path)
	tfidf_train_features, tfidf_test_features = get_tfidf_data(args.tf_path)
	cv_train_features, cv_test_features = get_count_vector(args.cv_path)
	#合并features
	#print d2v_train_features.shape
	#print tfidf_train_features.shape
	#print np.concatenate([d2v_train_features, tfidf_train_features], axis = 1).shape
	#train_data = feature_concat(d2v_train_features, tfidf_train_features)
	#test_data = feature_concat(d2v_test_features, tfidf_test_features)
	train_data = d2v_train_features
	test_data = d2v_test_features
	nb_classes = max(train_labels)+1
	#训练
	print 'train_labels: ',train_labels.shape
	gbm_model, lr_model, svm_model = train(train_data, train_labels)
	gbm_model.save_model('../models/gbm_model')
	gbm_y_pred = gbm_model.predict(test_data, num_iteration=gbm_model.best_iteration)
	print 'gbm_y_pred: ',gbm_y_pred.shape
	lr_y_pred = lr_model.predict_proba(test_data)
	print 'lr_y_pred: ',lr_y_pred.shape
	svm_y_pred = svm_model.predict_proba(test_data)
	print 'svm_y_pred: ',svm_y_pred.shape
	ensembled_pred = ensemble_prediction(gbm_y_pred, lr_y_pred, svm_y_pred)
	y_pred = one_hot_encode(ensembled_pred, nb_classes).astype(int)
	output(y_pred)

if __name__ == '__main__':
	main() 
