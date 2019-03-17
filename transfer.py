import os
import numpy
import sys
import torch
import json
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name):
    weights_path = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, model_name)
    model = torch.load(weights_path)
    return model

def judge_pure_english(keyword):
    return all(ord(c) < 128 for c in keyword)

def rev_wordpiece(str):
    #print(str)
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
            if i+1<len(str)-1 and str[i]=='\'':
                str[i] += str[i+1]
                str.remove(str[i+1])
        for i in range(len(str)-1, 0, -1):
            if not judge_pure_english(str[i]):
                str.remove(str[i])
    return str

def get_transfer_examples(data_dir, data_name):

    def _read_json(input_file):
        lines = []
        with open(input_file, "r") as f:
            for line in f:
                lines.append(json.loads(line))
            return lines
    fr = os.path.join(data_dir, data_name)
    lines = _read_json(fr)
    examples = []
    for (i, line) in enumerate(lines):
        text_a = line.get("line")
        mask_a = line.get("masks")
        label = line.get("label")
        examples.append([text_a, mask_a, label])
    return examples

# 不一定使用
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a raw_data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []

    for (ex_index, example) in enumerate(examples):
        text_a, mask_a, label = example
        tokens_a = tokenizer.tokenize(text_a)
        segment_id = label_map[label]
        masks = mask_a
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

        for index in masks:
            if index > max_seq_length - 1:
                break
            masked_lm_labels[index+1] = tokenizer.convert_tokens_to_ids([tokens[index+1]])[0]
            output_tokens[index+1] = "[MASK]"

        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        features.append([input_ids,input_mask,segment_ids,masked_lm_labels])
    return features

def convert_example_to_feature(example, tokenizer, out_tokens=False):
    """Loads a raw_data file into a list of `InputBatch`s."""
    text_a, mask_a, label = example
    tokens = []
    masks = []
    segment_id = 1 - int(label)
    tokens.append("[CLS]")
    for i, a in enumerate(text_a.split()):
        token_a = tokenizer.tokenize(a)
        if i in mask_a:
            mask_s = len(tokens)
            mask_e = len(token_a) + len(tokens)
            masks.extend(range(mask_s,mask_e))
            tokens.extend(["[MASK]"]*len(token_a))
        else:
            tokens.extend(token_a)
    tokens.append("[SEP]")
    segment_ids = [segment_id] * len(tokens)
    output_tokens = list(tokens)
    '''
    tokens_a = tokenizer.tokenize(text_a)
    segment_id = 1 - int(label)
    masks = mask_a

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

    output_tokens = list(tokens)
    masks = [m+1 for m in masks] # [cls]插入导致位移

    for index in masks:
        output_tokens[index] = "[MASK]"
    '''
    if out_tokens:
        return [tokens, output_tokens, segment_id]

    input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

    return [input_ids,masks,segment_ids, segment_id]


def main():
    config_file = sys.argv[1]
    step = sys.argv[2]
    with open(config_file, 'r') as f:
        configs_dict = json.load(f)

    task_name = configs_dict.get("task_name")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    if task_name == "yelp":
        if step == "bert_st":
            model = BertForMaskedLM.from_pretrained("{}/bert-base-uncased.tar.gz".format(PYTORCH_PRETRAINED_BERT_CACHE),cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
        elif step == "bert_ft":
            bert_name = "{}/BertForMaskedLM_yelp_wo_label_epoch10".format(task_name.lower())
            model = load_model(bert_name)
        elif step == "bert_ft_cls":
            bert_name = "{}/BertForMaskedLM_yelp_wo_label_w_cls_epoch10".format(task_name.lower())
            model = load_model(bert_name)
    model.cuda()
    model.eval()
    run_transfer(model, tokenizer, task_name, model_name="cbert")
    #delete_transfer(model, tokenizer, task_name)

def run_transfer(model, tokenizer, task_name, epoch=None, model_name=None, modified="", set="test"):
    data_dir = os.path.join(os.curdir, "processed_data" + modified, task_name + "/")
    train = get_transfer_examples(data_dir, "{}.data.label".format(set))
    if epoch:
        output_dir = os.path.join(os.curdir, "tranferred_data", task_name + "/")
    else:
        output_dir = os.path.join(os.curdir, "evaluation", "outputs", task_name + "/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if epoch:
        transferred_save = os.path.join(output_dir, "transffered.epoch_{}".format(epoch))
        test_save_0 = os.path.join(output_dir, "sentiment.{}.0".format(set))
        test_save_1 = os.path.join(output_dir, "sentiment.{}.1".format(set))
    else:
        test_save_0 = os.path.join(output_dir, "sentiment.{}.0.{}".format(set, model_name))
        test_save_1 = os.path.join(output_dir, "sentiment.{}.1.{}".format(set, model_name))
    if epoch:
        save_file = open(transferred_save, 'w')
    test_file_0 = open(test_save_0, 'w')
    test_file_1 = open(test_save_1, 'w')
    for example in train:
        ids, masks, segment_ids, cls = convert_example_to_feature(example, tokenizer)
        init_str = example[0]
        init_cls = 1 - cls
        if epoch:
            line_str = json.dumps({u'init_str': init_str, u'cls': str(init_cls)})
            save_file.write(line_str + '\n')
        ids_tensor = torch.tensor([ids])
        segment_tensors = torch.tensor([segment_ids])
        predictions = model(ids_tensor.cuda(), segment_tensors.cuda())
        for masked_index in masks:
            predicted_index = torch.argmax(predictions[0, masked_index]).item()
            ids[masked_index] = predicted_index
        tran_str = tokenizer.convert_ids_to_tokens(ids)
        tran_str = rev_wordpiece(tran_str)
        if epoch:
            line_str = json.dumps({u'tran_str': " ".join(tran_str), u'cls': str(cls)})
            save_file.write(line_str + '\n')
        if init_cls == 0:
            test_file_0.write(init_str + '\t' + " ".join(tran_str[1:-1]) + '\t' + "0" + '\n')
        elif init_cls == 1:
            test_file_1.write(init_str + '\t' + " ".join(tran_str[1:-1]) + '\t' + "1" + '\n')
    if epoch:
        save_file.close()
    test_file_0.close()
    test_file_1.close()


def delete_transfer(model, tokenizer, task_name):
    data_dir = os.path.join(os.curdir, "processed_data", task_name + "/")
    train = get_transfer_examples(data_dir, "test.data.label")
    output_dir = os.path.join(os.curdir, "evaluation", "outputs", task_name + "/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    test_save_0 = os.path.join(output_dir, "sentiment.test.0.delete")
    test_save_1 = os.path.join(output_dir, "sentiment.test.1.delete")

    test_file_0 = open(test_save_0, 'w')
    test_file_1 = open(test_save_1, 'w')
    for example in train:
        init_str, mask_str, cls = convert_example_to_feature(example, tokenizer, out_tokens=True)
        init_str = rev_wordpiece(init_str)
        mask_str = rev_wordpiece(mask_str)
        init_cls = 1 - cls

        if init_cls == 0:
            test_file_0.write(init_str + '\t' + " ".join(mask_str[1:-1]) + '\t' + "0" + '\n')
        elif init_cls == 1:
            test_file_1.write(init_str + '\t' + " ".join(mask_str[1:-1]) + '\t' + "1" + '\n')

    test_file_0.close()
    test_file_1.close()


if __name__ == "__main__":
    main()