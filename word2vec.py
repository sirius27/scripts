#coding:utf8
import gensim
import numpy as np
import tqdm

min_count=5
train_txt_path = '../temp_data/train_reduced.txt'
test_txt_path = '../temp_data/test_reduced.txt'

LINE_LENGTH = 60000
	    
with open(train_txt_path,'r') as f:
    train_lines = f.readlines()

with open(test_txt_path,'r') as f:
    test_lines = f.readlines()
#将train和test的文本合并训练词向量
sentences = train_lines + test_lines
model = gensim.models.Word2Vec(sentences, min_count=min_count)
#将train和test向量化
def lines_vectorize(lines, model):
    vecs = list()
    for line in tqdm.tqdm(lines):
        #一个sample
        line_vec = np.array([model[word].tolist() for word in line])
        if line_vec.shape[0] > LINE_LENGTH:
            line_vec = line_vec[-LINE_LENGTH:,:]
        elif line_vec.shape[0] < LINE_LENGTH:
            new_line_vec = np.zeros((LINE_LENGTH, line_vec.shape[1]))
            new_line_vec[-line_vec.shape[0]:,:]=line_vec
            line_vec=new_line_vec
        vecs.append(line_vec)
    return np.array(vecs)

train_vecs = lines_vectorize(train_lines, model)
test_vecs = lines_vectorize(test_lines, model)

np.save('../models/train_wv.npy',train_vecs)
np.save('../models/test_wv.npy',test_vecs)
