import argparse
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from torch.autograd import Variable
from tqdm import tqdm

from cnntext_wd import CNN
from utils import read_data, save_cls, load_cls

with open("run.config", 'rb') as f:
    configs_dict = json.load(f)

model_name = configs_dict.get("model_name")
task_name = configs_dict.get("task_name")
modified = configs_dict.get("modified")

def train(data, params):

    model = CNN(**params).cuda(params["GPU"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    for e in tqdm(range(params["EPOCH"])):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [sent for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        #if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
        #    print("early stopping by dev_acc!")
        #    break
        #else:
        #    pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [sent for sent in x]

    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="non-static", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="data", help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=10, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="the index of gpu to be used")

    options = parser.parse_args()
    if options.mode == "train":
        data, label_cnt = read_data(dir="raw_data/{}".format(task_name), train="train", dev="dev", test="test")
    else:
        data, label_cnt = read_data(dir="raw_data/{}".format(task_name), test="test")
    print(label_cnt)

    data["classes"] = sorted(list(set(data["train_y"])))

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        #"MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "MAX_SENT_LEN": 32,
        "BATCH_SIZE": 50,
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            save_cls(model, task_name, model_name)
            #save_vocab(data["vocab"], task_name, model_name)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = load_cls(task_name, model_name).cuda(params["GPU"])

        test_acc = test(data, model, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()
