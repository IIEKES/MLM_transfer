from sklearn.utils import shuffle

import pickle
import re
import numpy as np
import os

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower().split()

def read_test_data(dir="SST"):
	data = {}
	test_file_0 = "sentiment.test.0.cbert"
	test_file_1 = "sentiment.test.1.cbert"
	x, y = [], []
	with open(os.path.join(dir,test_file_0), "r", encoding="utf-8") as f:
		for line in f:
			if line[-1] == "\n":
				_, content, label = line.split('\t')
				y.append(1 - int(label))
				content = clean_str(content)
				x.append(content)
	with open(os.path.join(dir,test_file_1), "r", encoding="utf-8") as f:
		for line in f:
			if line[-1] == "\n":
				_, content, label = line.split('\t')
				y.append(1 - int(label))
				content = clean_str(content)
				x.append(content)
	x, y = shuffle(x, y)
	data["test_x"], data["test_y"] = x, y

	return data

def read_data(dir="SST", train=None, dev=None, test=None):

	def clean_str(string):
		"""
		Tokenization/string cleaning for all datasets except for SST.
		Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
		"""
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " \'s", string)
		string = re.sub(r"\'ve", " \'ve", string)
		string = re.sub(r"n\'t", " n\'t", string)
		string = re.sub(r"\'re", " \'re", string)
		string = re.sub(r"\'d", " \'d", string)
		string = re.sub(r"\'ll", " \'ll", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", " ! ", string)
		string = re.sub(r"\(", " \( ", string)
		string = re.sub(r"\)", " \) ", string)
		string = re.sub(r"\?", " \? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		return string.strip().lower().split()

	data = {}
	label_cnt = [0,0]
	if train:
		train_file_0 = "sentiment.train.0"
		train_file_1 = "sentiment.train.1"
		x, y = [], []
		with open(os.path.join(dir,train_file_0), "r", encoding="utf-8") as f:
			for line in f:
				if line[-1] == "\n":
					y.append(0)
					label_cnt[0] += 1
					line = clean_str(line)
					x.append(line)
		with open(os.path.join(dir,train_file_1), "r", encoding="utf-8") as f:
			for line in f:
				if line[-1] == "\n":
					y.append(1)
					label_cnt[1] += 1
					line = clean_str(line)
					x.append(line)
		x, y = shuffle(x, y)
		data["train_x"], data["train_y"] = x, y

	if dev:
		dev_file_0 = "sentiment.dev.0"
		dev_file_1 = "sentiment.dev.1"
		x, y = [], []
		with open(os.path.join(dir, dev_file_0), "r", encoding="utf-8") as f:
			for line in f:
				if line[-1] == "\n":
					y.append(0)
					line = clean_str(line)
					x.append(line)
		with open(os.path.join(dir, dev_file_1), "r", encoding="utf-8") as f:
			for line in f:
				if line[-1] == "\n":
					y.append(1)
					line = clean_str(line)
					x.append(line)
		x, y = shuffle(x, y)
		data["dev_x"], data["dev_y"] = x, y

	if test:
		test_file_0 = "sentiment.test.0"
		test_file_1 = "sentiment.test.1"
		x, y = [], []
		with open(os.path.join(dir,test_file_0), "r", encoding="utf-8") as f:
			for line in f:
				if line[-1] == "\n":
					y.append(0)
					line = clean_str(line)
					x.append(line)
		with open(os.path.join(dir,test_file_1), "r", encoding="utf-8") as f:
			for line in f:
				if line[-1] == "\n":
					y.append(1)
					line = clean_str(line)
					x.append(line)
		x, y = shuffle(x, y)
		data["test_x"], data["test_y"] = x, y

	return data, label_cnt

def save_cls(model, dataset, model_name):
	path = "pytorch_pretrained_cls/{}.{}.pkl".format(dataset, model_name)
	pickle.dump(model, open(path, "wb"))
	print("A model is saved successfully as {}!".format(path))

def save_vocab(vocab, dataset, model_name):
	path = "pytorch_pretrained_cls/{}_vocab.{}.pkl".format(dataset, m)
	pickle.dump(vocab, open(path, "wb"))
	print("A vocab is saved successfully as {}!".format(path))

def load_cls(dataset, model_name):
	path = "pytorch_pretrained_cls/{}.{}.pkl".format(dataset, model_name)

	try:
		model = pickle.load(open(path, "rb"))
		print("Model pytorch_pretrained_cls/{}.{}.pkl loaded successfully!".format(dataset, model_name))

		return model
	except:
		print("No available model such as {}!".format(path))
		exit()

def load_vocab(dataset, model_name):
	path = "pytorch_pretrained_cls/{}_vocab.{}.pkl".format(dataset, model_name)

	try:
		vocab = pickle.load(open(path, "rb"))
		print("Model pytorch_pretrained_cls/{}_vocab.{}.pkl loaded successfully!".format(dataset, model_name))

		return vocab
	except:
		print("No available vocab such as {}!".format(path))
		exit()
