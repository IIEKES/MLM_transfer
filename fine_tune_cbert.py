from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from tqdm import trange
import torch
from dataloader import load_data
import shutil, logging
from pytorch_pretrained_bert.modeling import BertForMaskedLM


from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from test_tools.li_test_tool.BLEU.my_bleu_evaluate import eval_bleu
from test_tools.li_test_tool.classify_Bilstm.my_acc_evaluate import eval_acc
from transfer import run_transfer

with open("run.config", 'rb') as f:
    configs_dict = json.load(f)

task_name = configs_dict.get("task_name")
model_name = configs_dict.get("model_name")
modified = configs_dict.get("modified")
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case="True")

def main():
    save_every_epoch=False

    args, train_dataloader, t_total, device, n_gpu = load_data()
    print("**********************************************************")
    print(args)
    # Prepare model
    #model = BertForMaskedLM.from_pretrained(args.bert_model,cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    def load_model(model_name):
        weights_path = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, model_name)
        model = torch.load(weights_path)
        return model
    bert_name = "{}/BertForMaskedLM_{}_epoch_10{}".format(task_name.lower(), task_name.lower(), modified)
    model = load_model(bert_name)
    model.to(device)

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

    # eval_bleu参数
    generate_file_0 = "evaluation/outputs/{}/sentiment.test.0.{}".format(task_name, model_name)
    dev_file_0 = "evaluation/outputs/{}/sentiment.dev.0.{}".format(task_name, model_name)
    orgin_file_0 = "evaluation/outputs/{}/sentiment.test.0.human".format(task_name)
    generate_file_1 = "evaluation/outputs/{}/sentiment.test.1.{}".format(task_name, model_name)
    dev_file_1 = "evaluation/outputs/{}/sentiment.dev.1.{}".format(task_name, model_name)
    orgin_file_1 = "evaluation/outputs/{}/sentiment.test.1.human".format(task_name)
    save_file_path = "evaluation/outputs/{}/{}_ft{}".format(task_name, model_name, modified)
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

    save_dict = {}
    acc_best = 0
    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, avg_loss, avg_acc = 0, 0, 0.
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            avg_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
            if (step + 1) % 50 == 0:
                print("avg_loss: {}".format(avg_loss / 50))
                avg_loss = 0

        run_transfer(model, tokenizer, task_name, model_name=model_name, modified=modified, set="dev")
        acc_0 = 1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                             test_rate=test_rate, input_file=dev_file_0)
        acc_1 = 1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                             test_rate=test_rate, input_file=dev_file_1)
        acc_avg = (acc_0 + acc_1) / 2
        print('{{"dev acc"：{}}}'.format(acc_avg))
        run_transfer(model, tokenizer, task_name, model_name=model_name, modified=modified)
        bleu_0 = eval_bleu(generate_file=generate_file_0, orgin_file=orgin_file_0)
        bleu_1 = eval_bleu(generate_file=generate_file_1, orgin_file=orgin_file_1)
        bleu_avg = (bleu_0 + bleu_1) / 2
        print('{{"bleu_0": {}, "bleu_1": {}, "bleu_avg": {}}}'.format(bleu_0, bleu_1, bleu_avg))
        acc_0 = 1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                             test_rate=test_rate, input_file=generate_file_0)
        acc_1 = 1 - eval_acc(dict_file=dict_file, train_rate=train_rate, valid_rate=valid_rate,
                             test_rate=test_rate, input_file=generate_file_1)
        acc_avg = (acc_0 + acc_1) / 2
        print('{{"acc_0": {}, "acc_1": {}, "acc_avg": {}}}'.format(acc_0, acc_1, acc_avg))

        if acc_avg > acc_best:
            acc_best = acc_avg
        if not (acc_avg in save_dict.keys() and save_dict[acc_avg] > bleu_avg):
            save_dict[acc_avg] = bleu_avg

        save_file_name_0 = os.path.join(save_file_path, "sentiment.test.0.{}.{}.{}.epoch_{}".format(model_name, acc_avg, bleu_avg, e))
        shutil.copy(generate_file_0, save_file_name_0)
        save_file_name_1 = os.path.join(save_file_path, "sentiment.test.1.{}.{}.{}.epoch_{}".format(model_name, acc_avg, bleu_avg, e))
        shutil.copy(generate_file_1, save_file_name_1)
        if save_every_epoch:
            save_model_name = "CBertForMaskedLM_" + task_name + "_epoch_" + str(e+1) + modified
            save_model_path = os.path.join(save_model_dir, save_model_name)
            torch.save(model, save_model_path)
        else:
            if (e+1) % 10 == 0:
                save_model_name = "CBertForMaskedLM_" + task_name + "_epoch_" + str(e+1) + modified
                save_model_path = os.path.join(save_model_dir, save_model_name)
                torch.save(model, save_model_path)
    print("Best result: acc {} bleu {}".format(acc_best, save_dict[acc_best]))
if __name__ == "__main__":
    main()