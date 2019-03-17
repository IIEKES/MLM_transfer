from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
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

# wuxing added
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
        if model_name == 'bert':
            segment_id = 0
        elif model_name == "cbert":
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

def load_data():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default="{}/bert-base-uncased.tar.gz".format(PYTORCH_PRETRAINED_BERT_CACHE),
                        type=str)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, )
    parser.add_argument("--max_seq_length", default=32, type=int)
    parser.add_argument("--do_train", default=True)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--no_cuda", default=False, action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--optimize_on_cpu', default=False, action='store_true')
    parser.add_argument('--loss_scale', type=float, default=128)

    args = parser.parse_args()

    args.data_dir = os.path.join(os.curdir, "processed_data" + modified, task_name + "/")
    args.output_dir = os.path.join("/tmp", task_name + "_output/")

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
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels(task_name)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=args.do_lower_case)

    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    train_examples.extend(dev_examples)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in train_features], dtype=torch.long)
    # forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_labels)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return args, train_dataloader, t_total, device, n_gpu