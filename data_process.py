#coding:utf8
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import time
import numpy as np
import pandas as pd

process = True
StopWordsLst=stopwords.words('english')
raw_train_text_path='../data/training_text'
raw_test_text_path='../data/testing_text'
raw_train_variants_path = '../data/training_variants'
raw_test_variants_path = '../data/testing_variants'

def pre_processing(text, center_lst):
    #reduce text
    #注意需要包括*
        #小写所有字母
    word_lst = [w.lower() for w in word_lst]
    #去掉停用词
    word_lst = [w for w in word_lst if not w in StopWordsLst]
    return word_lst

def reduce_text(lines, variations, genes, window_size):
	reduced_lines=[]
	processed_cnt=0
	for v,g,line in zip(variations, genes, lines):
		if processed_cnt > 1:
			break
		tokenizer = RegexpTokenizer('[a-zA-Z\*]+|[0-9\.]+&[^\.]')
		words = tokenizer.tokenize(line)
		
		# variants中带*的项，*只是指一个符号，还是类似于ls *的意思?
		# reduce text
		variant_centers = [i for i,word in enumerate(words) if word == v]
		#gene_centers= [i for i,word in enumerate(words) if word == g]
		#centers = variant_centers + gene_centers
		centers = variant_centers
		if len(centers) == 0:
			centers = np.arange(len(words))
		#这里肯定可以用map-reduce
		print centers
		print 'window size is ',window_size
		selected_intervals = map(lambda x:set(range(x-window_size, x+window_size)), centers)
		#print selected_intervals
		merged_interval=np.array(list(reduce(lambda x,y:x|y, selected_intervals)))
		merged_interval = list(set(merged_interval.clip(0,len(words)-1)))
		selected_pieces = [words[i] for i in merged_interval]

		selected_pieces=words
		#对选出来的区域进行预处理
		#最小化
		selected_pieces = [w.lower() for w in selected_pieces \
							if not w.lower() in StopWordsLst]
		new_reduced_line = ' '.join(selected_pieces)
		print 'original length %d, new length %d'%(len(words), len(selected_pieces))
		reduced_lines.append(new_reduced_line)
		if processed_cnt%100 == 0:
			print '%d lines processed'%processed_cnt
		processed_cnt+=1
	return reduced_lines

#获取训练text
def get_reduced_text(raw_text_path, raw_variants_path):
	with open(raw_text_path, 'r') as f:
		lines = f.readlines()
	lines = lines[1:]
	#获取variants和类别：
	variants = pd.read_csv(raw_variants_path, index_col=0)
	variations = variants['Variation'].values
	genes = variants['Gene'].values
	reduced_lines = reduce_text(lines, variations, genes, 5)
	output = '\n'.join(reduced_lines)
	return output


train_output = get_reduced_text(raw_train_text_path, raw_train_variants_path)
test_output = get_reduced_text(raw_test_text_path, raw_test_variants_path)
#最终得到的output.txt将被用于训练wordvector
train_output_path = '../temp_data/train_reduced.txt'
with open(train_output_path,'w') as f:
    f.write(train_output)
'''
test_output_path = '../temp_data/test_reduced.txt'
with open(test_output_path, 'w') as f:
	f.write(test_output)
'''
