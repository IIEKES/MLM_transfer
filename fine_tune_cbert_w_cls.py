from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
from test_tools.li_test_tool.BLEU.my_bleu_evaluate import eval_bleu
from test_tools.li_test_tool.classify_Bilstm.my_acc_evaluate import eval_acc
from test_tools.yang_test_tool.cnntext_wd import bert_embeddings, tokenizer
from test_tools.yang_test_tool.split import run_split
from test_tools.yang_test_tool.multi_bleu import eval_multi_bleu
from transfer import run_transfer
from utils import load_cls, read_test_data
from dataloader import load_data

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
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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
	
def main():
    save_every_epoch=False

    args, train_dataloader, t_total, device, n_gpu = load_data()
    print("**********************************************************")
    print(args)
    def load_model(model_name):
        weights_path = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, model_name)
        model = torch.load(weights_path)
        return model
    cbert_name = "{}/CBertForMaskedLM_{}_epoch_10{}".format(task_name.lower(), task_name.lower(), modified)
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
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0
        
    model.train()

    save_model_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, task_name)
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    cls_criterion = nn.CrossEntropyLoss()

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
    # eval_acc parameters
    dict_file = 'test_tools/li_test_tool/classify_Bilstm/data/style_transfer/zhi.dict.{}'.format(task_name)
    if task_name == 'yelp':
        train_rate = 0.9984
        valid_rate = 0.0008
        test_rate = 0.0008
    elif task_name == 'amazon':
        train_rate = 0.9989
        valid_rate = 0.00055
        test_rate = 0.00055

    acc_save_dict = {}
    bleu_save_dict = {}
    _acc_save_dict = {}
    _bleu_save_dict = {}
    count_dict = {}
    dev_acc_best = 0
    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, avg_loss, avg_acc = 0, 0, 0.
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            lm_loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
            segment_ids = 1 - segment_ids
            prediction_scores = model(input_ids, segment_ids, input_mask)
            prediction_scores = F.softmax(prediction_scores, dim=2)
            predicted_ids = prediction_scores @ bert_embeddings.weight
            batch_y = torch.stack([1-b[0] for b in batch[2]])
            pred, _ = cls(cls_model, predicted_ids, batch_y)
            # pred = F.softmax(pred, dim=1)
            cls_loss = cls_criterion(pred, batch_y)

            if lm_loss.item() > 1.5:
                loss = lm_loss / 100000 + cls_loss
            else:
                loss = cls_loss  # + lm_loss
            loss.backward()
            #tr_loss += cls_loss.item()
            avg_loss += cls_loss.item()
            #avg_acc += cls_acc
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if (step + 1) % 250 == 0:
                print("-------avg_loss: {}, lm_loss: {}--------".format(avg_loss / 250, lm_loss))
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
                if acc_avg > acc_threshold and dev_acc_avg > acc_threshold:
                    if not (dev_acc_avg < dev_acc_best):
                        # save_model_name = "BertForMaskedLM_" + task_name + "_acc_" + str(acc) + "w_cls"
                        # save_model_path = os.path.join(save_model_dir, save_model_name)
                        # torch.save(model, save_model_path)
                        save_file_name_0 = os.path.join(save_file_path,
                                                        "sentiment.test.0.{}.{}.{}".format(model_name, round(
                                                            acc_avg * 10) / 10.0, round(bleu_avg * 10) / 10.0))
                        shutil.copy(generate_file_0, save_file_name_0)
                        save_file_name_1 = os.path.join(save_file_path,
                                                        "sentiment.test.1.{}.{}.{}".format(model_name, round(
                                                            acc_avg * 10) / 10.0, round(bleu_avg * 10) / 10.0))
                        shutil.copy(generate_file_1, save_file_name_1)
                    if dev_acc_avg > dev_acc_best:
                        dev_acc_best = dev_acc_avg
                        acc_save_dict[dev_acc_avg] = acc_avg
                        bleu_save_dict[dev_acc_avg] = bleu_avg
                        _acc_save_dict[dev_acc_avg] = _acc
                        _bleu_save_dict[dev_acc_avg] = _bleu
                        count_dict[dev_acc_avg] = 1
                    elif dev_acc_avg == dev_acc_best:
                        acc_save_dict[dev_acc_avg] += acc_avg
                        bleu_save_dict[dev_acc_avg] += bleu_avg
                        _acc_save_dict[dev_acc_avg] += _acc
                        _bleu_save_dict[dev_acc_avg] += _bleu
                        count_dict[dev_acc_avg] += 1

        if save_every_epoch:
            save_model_name = "CBertForMaskedLM_" + task_name + "_w_cls_epoch" + str(e + 1) + modified
            save_model_path = os.path.join(save_model_dir, save_model_name)
            torch.save(model, save_model_path)
        else:
            if (e+1) % 10 == 0:
                save_model_name = "CBertForMaskedLM_" + task_name + "_w_cls_epoch" + str(e + 1) + modified
                save_model_path = os.path.join(save_model_dir, save_model_name)
                torch.save(model, save_model_path)
        cnt_best = count_dict[dev_acc_best]
        acc_best = round(acc_save_dict[dev_acc_best] * 10.0 / cnt_best) / 10.0
        bleu_best = round(bleu_save_dict[dev_acc_best] * 10.0 / cnt_best) / 10.0
        _acc_best = round(_acc_save_dict[dev_acc_best] * 10.0 / cnt_best) / 10.0
        _bleu_best = round(_bleu_save_dict[dev_acc_best] * 10.0 / cnt_best) / 10.0
        print("Best result: dev_acc {} acc {} bleu {} _acc {} _bleu {}".format(
            dev_acc_best, acc_best, bleu_best, _acc_best, _bleu_best))

if __name__ == "__main__":
    main()