from torch.autograd import Variable
import torch
import numpy as np
from utils import read_data, read_test_data, load_cls, load_vocab


def test_acc(model):
    data = read_test_data(dir="evaluation/outputs/yelp")

    x = data["test_x"]
    y = data["test_y"]

    model.eval()

    x = [sent for sent in x]


    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc

if __name__ == "__main__":
    cls = load_cls("yelp").cuda()
    print(test_acc(cls))