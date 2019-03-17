import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.autograd import Variable
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case="True")

with open("run.config", 'rb') as f:
    configs_dict = json.load(f)

task_name = configs_dict.get("task_name")

bert_model = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "bert-base-uncased.tar.gz")

def load_model(model_name):
    weights_path = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, model_name)
    model = torch.load(weights_path)
    return model

model = BertForMaskedLM.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.cuda()
model.eval()
bert_embeddings = model.bert.embeddings.word_embeddings
bert_embeddings.weight.requires_grad = False


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

class EncoderRNN(nn.Module):
    def __init__(self, h_dim, gpu=True,  batch_first=True):
        super(EncoderRNN, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.emb_dim = bert_embeddings.weight.shape[-1]
        self.lstm = nn.LSTM(self.emb_dim, h_dim, batch_first=batch_first,
                            bidirectional=True)
        self.MAX_SENT_LEN = 32

        for m in self.modules():
            # print(m.__class__.__name__)
            weights_init(m)


    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1 * 2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1 * 2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def embedding(self, inp, ignore_step=False):
        if ignore_step:
            return inp
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
                words_embedding = Variable(torch.FloatTensor(numpy.zeros((self.MAX_SENT_LEN, self.emb_dim)))).cuda()
                words_embeddings.append(words_embedding)
        words_embeddings = torch.stack(words_embeddings)
        return words_embeddings

    '''
    def embedding(self, inp, ignore_step=False):
        if ignore_step:
            return inp
        input_ids = []
        for example in inp:
            output_tokens = [tokenizer.tokenize(e)[0] for e in example]
            ids = tokenizer.convert_tokens_to_ids(output_tokens)
            while len(ids) < self.MAX_SENT_LEN:
                ids.append(0)
            input_ids.append(ids[:self.MAX_SENT_LEN])
        input_ids = Variable(torch.LongTensor(input_ids)).cuda()
        masks = input_ids != 0
        words_embeddings = bert_embeddings(input_ids)
        return masks, words_embeddings
    '''
    def forward(self, sentence, ignore_step=False):
        emb = self.embedding(sentence, ignore_step)
        self.hidden = self.init_hidden(emb.size(0))
        packed_emb = emb
        out, hidden = self.lstm(packed_emb, self.hidden)
        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]

        return out


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24, 1)
        )


    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.view(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)


class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(h_dim)
        self.main = nn.Linear(h_dim, c_num)
        for m in self.modules():
            #print(m.__class__.__name__)
            weights_init(m)

    def forward(self, encoder_outputs):
        attns = self.attn(encoder_outputs)  # (b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        return self.main(feats), attns

class RNNAttnCls(nn.Module):
    def __init__(self, **kwargs):
        super(RNNAttnCls, self).__init__()
        self.h_dim = kwargs["H_DIM"]
        self.class_size = kwargs["CLASS_SIZE"]
        self.encoder = EncoderRNN(self.h_dim).cuda()
        self.classifier = AttnClassifier(self.h_dim, self.class_size).cuda()

    def forward(self, sentence, ignore_step=False):
        encoder_outputs = self.encoder(sentence, ignore_step)
        output, attn = self.classifier(encoder_outputs)
        return output, attn