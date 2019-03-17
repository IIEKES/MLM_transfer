import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.autograd import Variable
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case="True")

with open("run.config", 'rb') as f:
	configs_dict = json.load(f)

model_name = configs_dict.get("model_name")
task_name = configs_dict.get("task_name")
modified = configs_dict.get("modified")


def load_model(model_name):
	weights_path = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, model_name)
	model = torch.load(weights_path)
	return model

if model_name == "cbert":
	cbert_name = "{}/CBertForMaskedLM_{}_epoch_10{}".format(task_name.lower(), task_name.lower(),modified)
	model = load_model(cbert_name)

model.cuda()
model.eval()
bert_embeddings = model.bert.embeddings.word_embeddings
bert_embeddings.weight.requires_grad = False

class CNN(nn.Module):
	def __init__(self, **kwargs):
		super(CNN, self).__init__()

		self.MODEL = kwargs["MODEL"]
		self.BATCH_SIZE = kwargs["BATCH_SIZE"]
		self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
		self.WORD_DIM = bert_embeddings.weight.shape[-1]
		self.CLASS_SIZE = kwargs["CLASS_SIZE"]
		self.FILTERS = kwargs["FILTERS"]
		self.FILTER_NUM = kwargs["FILTER_NUM"]
		self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
		self.IN_CHANNEL = 1

		assert (len(self.FILTERS) == len(self.FILTER_NUM))

		for i in range(len(self.FILTERS)):
			conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
			setattr(self, 'conv_{}'.format(i), conv)

		self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

	def get_conv(self, i):
		return getattr(self, 'conv_{}'.format(i))
	'''
	def embedding(self, inp, label_ids=None):
		if label_ids is not None:
			words_embeddings = []
			for i in range(inp.shape[0]):
				words_embedding = inp[i]
				ids = label_ids[i] == -1
				masks = torch.nonzero(ids)
				masks = torch.squeeze(masks)
				pad = torch.FloatTensor(numpy.zeros(words_embedding.shape)).cuda()
				pad = torch.index_select(pad, 0, masks)
				words_embedding.index_copy_(0, masks, pad)
				words_embeddings.append(words_embedding)
			words_embeddings = torch.stack(words_embeddings)
			return words_embeddings
		# input_ids = []
		words_embeddings = []
		for example in inp:
			output_tokens = [tokenizer.tokenize(e)[0] for e in example]
			if len(output_tokens) > 0:
				ids = tokenizer.convert_tokens_to_ids(output_tokens)
				if len(ids) > self.MAX_SENT_LEN:
					ids = ids[:self.MAX_SENT_LEN]
				ids = Variable(torch.LongTensor(ids)).cuda()
				words_embedding = bert_embeddings(ids)
				if len(ids) < self.MAX_SENT_LEN:
					pad_len = self.MAX_SENT_LEN - len(ids)
					emb_size = words_embedding.shape[1]
					pad = Variable(torch.FloatTensor(numpy.zeros((pad_len, emb_size)))).cuda()
					words_embedding = torch.cat((words_embedding, pad), 0)
				words_embeddings.append(words_embedding)
			else:
				print(example)
		words_embeddings = torch.stack(words_embeddings)
		return words_embeddings
	'''
	def embedding(self, inp, ignore_step=False, ignore_tokenize=False):
		if ignore_step:
			return inp
		input_ids = []
		if ignore_tokenize:
			words_embeddings = bert_embeddings(inp)
		else:
			for example in inp:
				output_tokens = [tokenizer.tokenize(e)[0] for e in example]
				ids = tokenizer.convert_tokens_to_ids(output_tokens)
				while len(ids) < self.MAX_SENT_LEN:
					ids.append(0)
				input_ids.append(ids[:self.MAX_SENT_LEN])
			input_ids = Variable(torch.LongTensor(input_ids)).cuda()
			words_embeddings = bert_embeddings(input_ids)
		return words_embeddings

	def forward(self, inp, ignore_step=False, ignore_tokenize=False):
		x = self.embedding(inp, ignore_step,ignore_tokenize).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
		if self.MODEL == "multichannel":
			x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
			x = torch.cat((x, x2), 1)

		conv_results = [
			F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] - 1)
				.view(-1, self.FILTER_NUM[i])
			for i in range(len(self.FILTERS))]

		x = torch.cat(conv_results, 1)
		x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
		x = self.fc(x)

		return x