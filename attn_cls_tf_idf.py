from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F
from utils import read_data, clean_str, load_cls, load_vocab
from functools import cmp_to_key
import sys
import os
import json

preprocessed_data_dir=sys.argv[1]
batch_size = 16
max_line = 500000

with open("run.config", 'rb') as f:
    configs_dict = json.load(f)

model_name = configs_dict.get("model_name")
task_name = configs_dict.get("task_name")
modified = configs_dict.get("modified")

def cmp(a, b):
    return (a>b)-(a<b)

def cls_tf_idf(model, label):
    save_path = os.path.join(preprocessed_data_dir, task_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fr = open(os.path.join(save_path, "{}.train.{}.tf_idf.filter.label".format(task_name, label)), 'r')
    lines = []
    scores = []
    for l in fr:
        line, score = l.split('\t')
        lines.append(line)
        scores.append(score)
    fr.close()

    lines = lines[:20000]
    scores = scores[:20000]
    fw = open(os.path.join(preprocessed_data_dir, "{}/{}.train.{}.tf_idf.attn.label".format(task_name, task_name, label)), 'w')
    tf_idf = []
    line_num = min(len(lines), max_line)
    for i in range(0, line_num, batch_size):
        batch_range = min(batch_size, line_num - i)
        batch_lines = lines[i:i + batch_range]
        batch_x = [clean_str(sent) for sent in lines[i:i + batch_range]]
        batch_scores = [float(score) for score in scores[i:i + batch_range]]
        pred, _ = model(batch_x)
        batch_mu= F.sigmoid(pred)
        for x, line, score, mu in zip(batch_x, batch_lines, batch_scores, batch_mu):
            if len(x) > 0:
                mu = mu[label].item()
                score = mu * score
                #score = mu / len(x)
                #score = mu
                tf_idf.append([line, score])
    tf_idf.sort(key=cmp_to_key(lambda a, b: b[1] - a[1]))
    for i in tf_idf:
        fw.write(i[0] + '\t' + str(i[1]) + '\n')
    fw.close()
    print("processed over!")

if __name__ == "__main__":
    cls = load_cls("{}".format(task_name), "attn.{}".format(model_name)).cuda()
    for i in cls.parameters():
        i.requires_grad = False
    cls.eval()
    cls_tf_idf(cls, 1)
    cls_tf_idf(cls, 0)