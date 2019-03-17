from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
from test_tools.li_test_tool.BLEU.my_bleu_evaluate import eval_bleu
from test_tools.li_test_tool.classify_Bilstm.my_acc_evaluate import eval_acc
from test_tools.yang_test_tool.cnntext_wd import tokenizer
from test_tools.yang_test_tool.split import run_split
from test_tools.yang_test_tool.multi_bleu import eval_multi_bleu
from transfer import run_transfer
from utils import load_cls, read_test_data

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

with open("run.config", 'rb') as f:
    configs_dict = json.load(f)

model_name = configs_dict.get("model_name")
task_name = configs_dict.get("task_name")
modified = configs_dict.get("modified")
acc_threshold = configs_dict.get("acc_threshold")

class DataProcessor(object):
    """Base class for raw_data converters for sequence classification raw_data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this raw_data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, "r") as f:
            for line in f:
                lines.append(json.loads(line))
            return lines

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, mask_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.guid = guid
        self.text_a = text_a
        self.mask_a = mask_a
        self.label = label


class InputFeatures(object):
    """A single set of features of raw_data."""

    def __init__(self, input_ids, input_mask, segment_ids, masked_lm_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels

# 伍星added
class biLabelProcessor(DataProcessor):
    """Processor for the CoLA raw_data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.data.label")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.data.label")), "dev")

    def get_labels(self, name):
        """See base class."""
        if name in ['yelp', 'amazon', 'imagecaption']:
            return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line.get("line")
            mask_a = line.get("masks")
            label = line.get("label")
            examples.append(
                InputExample(guid=guid, text_a=text_a, mask_a=mask_a, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a raw_data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        segment_id = label_map[example.label]
        masks = example.mask_a
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        # 由于是CMLM，所以需要用标签
        tokens = []
        segment_ids = []
        # 是不是可以去掉[CLS]和[SEP]
        tokens.append("[CLS]")
        segment_ids.append(segment_id)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(segment_id)
        tokens.append("[SEP]")
        segment_ids.append(segment_id)
        masked_lm_labels = [-1] * max_seq_length

        output_tokens = list(tokens)
        #print(tokens)
        for index in masks:
            if index + 1 > max_seq_length - 1:
                break
            #print(index+1)
            masked_lm_labels[index+1] = tokenizer.convert_tokens_to_ids([tokens[index+1]])[0]
            output_tokens[index+1] = "[MASK]"

        init_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0) # ?segment_id

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("init_ids: %s" % " ".join([str(x) for x in init_ids]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              masked_lm_labels=masked_lm_labels))
    return features

def main():
    parser = argparse.ArgumentParser()
    ## BERT Config
    parser.add_argument("--bert_model", default="{}/bert-base-uncased.tar.gz".format(PYTORCH_PRETRAINED_BERT_CACHE), type=str)
    parser.add_argument("--task_name",default=None,type=str)
    parser.add_argument("--output_dir",default=None,type=str,)
    parser.add_argument("--max_seq_length",default=32,type=int)
    parser.add_argument("--do_eval",default=True)
    parser.add_argument("--do_lower_case",default=True)
    parser.add_argument("--train_batch_size",default=32,type=int)
    parser.add_argument("--eval_batch_size",default=8,type=int)
    parser.add_argument("--learning_rate",default=2e-5,type=float)
    parser.add_argument("--test_epoch",default=10,type=int)
    parser.add_argument("--warmup_proportion",default=0.1, type=float)
    parser.add_argument("--no_cuda",default=False,action='store_true')
    parser.add_argument("--local_rank",type=int,default=-1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1)
    parser.add_argument('--optimize_on_cpu',default=False,action='store_true')
    parser.add_argument('--loss_scale',type=float, default=128)

    args = parser.parse_args()
    args.task_name = task_name
    args.data_dir = os.path.join(os.curdir, "processed_data"+modified, task_name + "/")
    args.output_dir = os.path.join("/tmp", task_name + "_output/")
    #print("**********************************************************")
    #print(args)
    run_aug(args, save_every_epoch=False)

def cls(model, x, y):
    pred = model(x, True)
    pred_y = np.argmax(pred.cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred_y, y)]) / len(pred_y)
    return pred, acc

def cls_test(model, task_name):
    data = read_test_data(dir="evaluation/outputs/{}".format(task_name))

    x = data["test_x"]
    y = data["test_y"]
    x = [sent for sent in x]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def run_aug(args, save_every_epoch=False):

    processors = {
        "yelp": biLabelProcessor,
        "amazon": biLabelProcessor,
        "imagecaption": biLabelProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels(task_name)


    def load_model(model_name):
        weights_path = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, model_name)
        model = torch.load(weights_path)
        return model
    cbert_name = "{}/CBertForMaskedLM_{}_epoch_{}{}".format(task_name.lower(), task_name.lower(), args.test_epoch, modified)
    model = load_model(cbert_name)
    model.to(device)

    cls_model = load_cls(task_name, model_name).cuda()
    for i in cls_model.parameters():
        i.requires_grad = False
    cls_model.eval()

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']


    if args.do_eval:

        # eval_bleu参数
        generate_file_0 = "evaluation/outputs/{}/sentiment.test.0.{}".format(task_name, model_name)
        dev_file_0 = "evaluation/outputs/{}/sentiment.dev.0.{}".format(task_name, model_name)
        orgin_file_0 = "evaluation/outputs/{}/sentiment.test.0.human".format(task_name)
        generate_file_1 = "evaluation/outputs/{}/sentiment.test.1.{}".format(task_name, model_name)
        dev_file_1 = "evaluation/outputs/{}/sentiment.dev.1.{}".format(task_name, model_name)
        orgin_file_1 = "evaluation/outputs/{}/sentiment.test.1.human".format(task_name)
        save_file_path = "evaluation/outputs/{}/{}_ft_wc{}".format(task_name, model_name, modified)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        # eval_acc参数
        dict_file = 'test_tools/li_test_tool/classify_Bilstm/data/style_transfer/zhi.dict.{}'.format(task_name)
        if task_name == 'yelp':
            train_rate = 0.9984
            valid_rate = 0.0008
            test_rate = 0.0008
        elif task_name == 'amazon':
            train_rate = 0.9989
            valid_rate = 0.00055
            test_rate = 0.00055

        run_transfer(model, tokenizer, task_name, model_name=model_name, modified=modified, set="dev")
        dev_acc_0 = 1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                                 test_rate=test_rate, input_file=dev_file_0)
        dev_acc_1 = 1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                                 test_rate=test_rate, input_file=dev_file_1)
        dev_acc_avg = (dev_acc_0 + dev_acc_1) / 2
        dev_acc_avg = round(dev_acc_avg * 1000) / 10.0
        print('{{"dev acc"：{}}}'.format(dev_acc_avg))
        avg_loss = 0
        run_transfer(model, tokenizer, task_name, model_name=model_name, modified=modified)
        bleu_0 = eval_bleu(generate_file=generate_file_0, orgin_file=orgin_file_0) * 100
        bleu_1 = eval_bleu(generate_file=generate_file_1, orgin_file=orgin_file_1) * 100
        bleu_avg = (bleu_0 + bleu_1) / 2
        print('{{"bleu_0": {}, "bleu_1": {}, "bleu_avg": {}}}'.format(bleu_0, bleu_1,
                                                                      round(bleu_avg * 10) / 10.0))
        acc_0 = (1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                              test_rate=test_rate, input_file=generate_file_0)) * 100
        acc_1 = (1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                              test_rate=test_rate, input_file=generate_file_1)) * 100
        acc_avg = (acc_0 + acc_1) / 2
        print('{{"acc_0": {}, "acc_1": {}, "acc_avg": {}}}'.format(acc_0, acc_1,
                                                                   round(acc_avg* 10) / 10.0))
        _acc = cls_test(cls_model, task_name) * 100
        run_split(generate_file_0)
        run_split(generate_file_1)
        _bleu = eval_multi_bleu(model_name, task_name)
        print('{{"_ACCU": {}, "_BLEU": {}}}'.format(round(_acc * 10) / 10.0, round(_bleu * 10) / 10.0))

if __name__ == "__main__":
    main()