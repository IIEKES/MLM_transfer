from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from tqdm import trange
import torch
import logging
from dataloader import load_data

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

with open("run.config", 'rb') as f:
    configs_dict = json.load(f)
task_name = configs_dict.get("task_name")
model_name = configs_dict.get("model_name")
modified = configs_dict.get("modified")
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    save_every_epoch=False

    args, train_dataloader, t_total, device, n_gpu = load_data()
    print("**********************************************************")
    print(args)
    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model,cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
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

        if save_every_epoch:
            save_model_name = "BertForMaskedLM_" + task_name + "_epoch_" + str(e+1) + modified
            save_model_path = os.path.join(save_model_dir, save_model_name)
            torch.save(model, save_model_path)
        else:
            if (e+1) % 10 == 0:
                save_model_name = "BertForMaskedLM_" + task_name + "_epoch_" + str(e+1) + modified
                save_model_path = os.path.join(save_model_dir, save_model_name)
                torch.save(model, save_model_path)

if __name__ == "__main__":
    main()