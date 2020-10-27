import os
import copy
import json
import logging
from itertools import islice

from tqdm.notebook import tqdm
import pandas as pd
import regex as re
import torch
from torch.utils.data import TensorDataset

from utils import get_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, brands):
        self.guid = guid
        self.words = words
        self.brands = brands

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NaverNerProcessor(object):
    """Processor for the Naver NER data set """

    def __init__(self, args):
        self.args = args
        self.labels_lst = get_labels(args)

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            

            return lines
        

    def clean_review(self,review):
        review = ''.join(re.compile('[가-힣.!?a-zA-Z ]').findall(review))
        review = re.sub('\n', ' ', review)
        review = re.sub(r'\!+', '!', review)
        review = re.sub(r'\?+', '?', review)
        review = re.sub(r'\.+', '.', review)
        review = re.sub(r'\([^)]*\)', '', review)    # 괄호 안 내용 삭제
        review = re.sub(r'\[[^)]*\]', '', review)    # 대괄호 안 내용 삭제
        return review 
        
    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        
        examples = []
        
        
        for (i, data) in enumerate(dataset):
            try:
                brands, words = data.split('\t')
            except:
                continue
            #print(brands, words)
            #words = words.split()
            guid = "%s-%s" % (set_type, i)
            

            if i % 10000 == 0:
                logger.info(data)
            examples.append(InputExample(guid=guid, words=words, brands=brands))
        
        
        
        
        #for (i, data) in enumerate(dataset):
        #    words, labels = data.split('\t')
        #    words = self.clean_review(words)
        #    
        #    words = words.split()
        #    labels = labels.split()
        #    guid = "%s-%s" % (set_type, i)

            #labels_idx = []
            #for label in labels:
            #    labels_idx.append(self.labels_lst.index(label) if label in self.labels_lst else self.labels_lst.index("UNK"))
                
        #    print(words, len(words), labels_idx, len(labels_idx))
        #    assert len(words) == len(labels_idx)

        #    if i % 10000 == 0:
        #        logger.info(data)
        #    examples.append(InputExample(guid=guid, words=words, labels=labels_idx))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {
    "naver-ner": NaverNerProcessor,
}


def find(target, seq):
    max_window_size = len(target)
    if max_window_size <= 0:
        max_window_size = 1
        
    def window(seq, n=max_window_size):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
      
    
    def tune_window_size():
        window_size = max_window_size
        for i in range(max_window_size, 0, -1):
      
            for _s in window(seq=seq, n=i):# _s : 윈도우 결과
                if target in ''.join(_s):
                    window_size = i
                    #print(i, _s, window_size)
                    break
                    
        return window_size
                
    window_size = tune_window_size()
      
        
    def get_ckpts(_seq):
        pos = 0
        ckpt = list()
          
        for _s in window(seq=_seq, n=window_size):
            if target in ''.join(_s):
                ckpt.append([pos, pos+window_size])
            pos += 1
        return ckpt
            
    ckpts = get_ckpts(seq)
    label_list = ['O']*len(seq)
    
    for ckpt in ckpts:
        label_list[ckpt[0]] = 'brand-B'
        for i in range(ckpt[0]+1, ckpt[1]):
            label_list[i] = 'brand-I'
    
  
      
    
    return label_list

def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    
    labels_lst = [label.strip() for label in open(os.path.join('data', 'label.txt'), 'r', encoding='utf-8')]

    features = []
    i = 0
    for (ex_index, example) in enumerate(examples):
        
        
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        #print(example.words, example.brands)
        # Tokenize word by word (for NER)
        tokens = []
        label_ids = []
        
        
        
        
        
        words = example.words.split()
        
        for word in words:
            tokens.extend(tokenizer.tokenize(word))
        
        #tokens = tokenizer.tokenize(example.words)
        

        
        
        
        
        
        labels = find(example.brands, tokens)
        
        assert len(tokens) == len(labels)
        
        
        ###0914 수정 브랜드 토큰이 있는 것만 사용
        ###0915 브랜드 없어도 20퍼센트는 사용
        if 'brand-B' not in labels:
            i += 1
            if i < 2:
                pass
            elif i >= 9:
                i = 0
                continue
            else: 
                continue
        ###
        
        #if 'brand-B' not in labels:
        #    continue
        
        label_ids = []
        for label in labels:
            label_ids.append(labels_lst.index(label) if label in labels_lst else labels_lst.index("UNK"))
            
        if not tokens:
            tokens = [unk_token]  # For handling the bad-encoded word
            
            
        #print(example.brands, tokens, labels, label_ids)
        
        #print(word_tokens, slot_label)
        
        
        
        #for word in example.words:
        #    word_tokens = tokenizer.tokenize(word)
        #    labels = find(example.brands, word_tokens)
        #    
        #    
        #    slot_label = []
        #    for label in labels:
        #        slot_label.append(labels_lst.index(label) if label in labels_lst else labels_lst.index("UNK"))
        #        
        #    #print("asdfasdfdas:",word_tokens,word, slot_label)
        #    
        #    
        #    if not word_tokens:
        #        word_tokens = [unk_token]  # For handling the bad-encoded word
        #        
        #    tokens = word_tokens
        #    label_ids = slot_label
        #    #tokens.extend(word_tokens)
        #    ## Use the real label id for the first token of the word, and padding ids for the remaining tokens
        #    #label_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
        #    print('123123123',tokens, example.brands, label_ids)

        
        
            
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            label_ids = label_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        label_ids = label_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(label_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(label_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    return dataset
