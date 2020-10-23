#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import logging
import argparse
import random
import unicodedata
import shutil
from itertools import islice

import regex as re
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm, trange


from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification
)
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


# In[2]:


SPIECE_UNDERLINE = u'▁'


class KoBertTokenizer(PreTrainedTokenizer):
    """
        SentencePiece based tokenizer. Peculiarities:
            - requires `SentencePiece <https://github.com/google/sentencepiece>`_
    """
    vocab_files_names = {"vocab_file": "tokenizer_78b3253a26.model",
                     "vocab_txt": "vocab.txt"}
    
    pretrained_vocab_files_map = {
            "vocab_file": {
                "monologg/kobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert/tokenizer_78b3253a26.model",
                "monologg/kobert-lm": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert-lm/tokenizer_78b3253a26.model",
                "monologg/distilkobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/distilkobert/tokenizer_78b3253a26.model"
            },
            "vocab_txt": {
                "monologg/kobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert/vocab.txt",
                "monologg/kobert-lm": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/kobert-lm/vocab.txt",
                "monologg/distilkobert": "https://s3.amazonaws.com/models.huggingface.co/bert/monologg/distilkobert/vocab.txt"
            }
        }
    
    pretrained_init_configuration = {
            "monologg/kobert": {"do_lower_case": False},
            "monologg/kobert-lm": {"do_lower_case": False},
            "monologg/distilkobert": {"do_lower_case": False}
        }

    max_model_input_sizes = {
            "monologg/kobert": 512,
            "monologg/kobert-lm": 512,
            "monologg/distilkobert": 512
        }

    def __init__(
            self,
            vocab_file,
            vocab_txt,
            do_lower_case=False,
            remove_space=True,
            keep_accents=False,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

        # Build vocab
        self.token2idx = dict()
        self.idx2token = []
        with open(vocab_txt, 'r', encoding='utf-8') as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.token2idx[token] = idx
                self.idx2token.append(token)

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                           "pip install sentencepiece")

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.vocab_txt = vocab_txt

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.idx2token)

    def get_vocab(self):
        return dict(self.token2idx, **self.added_tokens_encoder)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use KoBertTokenizer: https://github.com/google/sentencepiece"
                           "pip install sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, return_unicode=True, sample=False):
        """ Tokenize a string. """
        text = self.preprocess_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def _convert_id_to_token(self, index, return_unicode=True):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.idx2token[index]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A KoBERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model
        Returns:
            A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A KoBERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return

        # 1. Save sentencepiece model
        out_vocab_model = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_model):
            shutil.copyfile(self.vocab_file, out_vocab_model)

        # 2. Save vocab.txt
        index = 0
        out_vocab_txt = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_txt"])
        with open(out_vocab_txt, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(out_vocab_txt)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

        return out_vocab_model, out_vocab_txt


# In[3]:


MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',
}


# In[4]:


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
        self.labels_lst = args.label_lst

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
            guid = "%s-%s" % (set_type, i)
            

            if i % 10000 == 0:
                logger.info(data)
            examples.append(InputExample(guid=guid, words=words, brands=brands))

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





# In[5]:


class NER:
    def __init__(self):
        self.set_ner_config()
        self.init_logger()
        self.set_seed(self.ner_config)
        self.tokenizer = self.load_tokenizer(self.ner_config)
        
        
    def init_logger(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)    

    def set_ner_config(self):
        def set_flags(**args):
            flag = SimpleNamespace(**args)
            return flag
        
        self.ner_config = set_flags(no_cuda = None
                                    ,task = 'naver-ner'
                                    ,model_dir = './model'#'root/analysis/product_cluster/model'
                                    ,data_dir = 'root/analysis/product_cluster/data'
                                    ,pred_dir = 'root/analysis/product_cluster/pred'
                                    ,output_file = 'distil_30epochs_4.txt'
                                    ,batch_size = 32
                                    ,label_lst = ['UNK', 'O', 'brand-B', 'brand-I']
                                    ,seed = 42
                                    ,model_type = 'distilkobert'
                                    
                                    #*** training args ***#
                                    ,train_file = 'train_01.tsv'
                                    ,train_batch_size = 64
                                    ,num_train_epochs = 30.0
                                    ,learning_rate = 5e-5
                                    ,max_seq_len = 50
                                    ,save_steps = 1000
                                    ,logging_steps = 50
                                    
                                    ,weight_decay = 0.0
                                    ,gradient_accumulation_steps = 1
                                    ,adam_epsilon = 1e-8
                                    ,max_grad_norm = 1.0
                                    ,max_steps = -1
                                    ,warmup_steps = 0
                                    
                                    
                                    #*** eval args ***#
                                    ,do_eval = True
                                    ,test_file = 'test_01.tsv'
                                    ,eval_batch_size = 64
                                    ,write_pred = True
                                   )
        
        self.ner_config.model_name_or_path = MODEL_PATH_MAP[self.ner_config.model_type]
        
    def set_seed(self, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not args.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)    
        
    def load_tokenizer(self,args):
        return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)    
        
    def do_train(self):
        
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

            labels_lst = self.ner_config.label_lst #[label.strip() for label in open(os.path.join('data', 'label.txt'), 'r', encoding='utf-8')]

            features = []
            i = 0
            for (ex_index, example) in enumerate(examples):
                if ex_index % 5000 == 0:
                    logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                tokens = []
                label_ids = []

                words = example.words.split()

                for word in words:
                    tokens.extend(tokenizer.tokenize(word))

                labels = find(example.brands, tokens)
                assert len(tokens) == len(labels)

                if 'brand-B' not in labels:
                    i += 1
                    if i < 2:
                        pass
                    elif i >= 9:
                        i = 0
                        continue
                    else: 
                        continue

                label_ids = []
                for label in labels:
                    label_ids.append(labels_lst.index(label) if label in labels_lst else labels_lst.index("UNK"))

                if not tokens:
                    tokens = [unk_token]  # For handling the bad-encoded word

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
        
        
        
        
        self.train_dataset = load_and_cache_examples(self.ner_config, self.tokenizer, mode="train")
        self.test_dataset = load_and_cache_examples(self.ner_config, self.tokenizer, mode="test")
        
        self.label_lst = self.ner_config.label_lst
        self.num_labels = len(self.label_lst)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[self.ner_config.model_type]

        self.config = self.config_class.from_pretrained(self.ner_config.model_name_or_path,
                                                        num_labels=self.num_labels,
                                                        finetuning_task=self.ner_config.task,
                                                        id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                        label2id={label: i for i, label in enumerate(self.label_lst)})
        self.model = self.model_class.from_pretrained(self.ner_config.model_name_or_path, config=self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not self.ner_config.no_cuda else "cpu"
        self.model.to(self.device)

        self.test_texts = None
        if self.ner_config.write_pred:
            # Empty the original prediction files
            if os.path.exists(self.ner_config.pred_dir):
                shutil.rmtree(self.ner_config.pred_dir)
        self.train()
        
        
        
    
    def do_predict(self, input_source):
        brands = self.predict(input_source)
        brands = [re.sub('▁','',b) for b in brands]
        return brands
    
    
    def get_device(self):
        return "cuda" if torch.cuda.is_available() and not self.ner_config.no_cuda else "cpu"


    def get_args(self):
        return torch.load(os.path.join(self.ner_config.model_dir, 'training_args.bin'))


    def load_model(self, args, device):
        # Check whether model exists
        if not os.path.exists(self.ner_config.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            model = AutoModelForTokenClassification.from_pretrained(self.ner_config.model_dir)  # Config will be automatically loaded from model_dir
            model.to(device)
            model.eval()
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

        return model

    def clean_reivew(self,review):
        review = ''.join(re.compile('[가-힣.!?a-zA-Z ]').findall(review))
        review = re.sub('\n', ' ', review)
        review = re.sub(r'\!+', '!', review)
        review = re.sub(r'\?+', '?', review)
        review = re.sub(r'\.+', '.', review)
        review = re.sub(r'\([^)]*\)', '', review)    # 괄호 안 내용 삭제
        review = re.sub(r'\[[^)]*\]', '', review)    # 대괄호 안 내용 삭제
        return review 

    def read_input_file(self, tokenizer):

        lines = []
        texts = []

        for line in self.ner_config.input_source:

            line = self.clean_reivew(line)

            texts.append(line)

            line = tokenizer.tokenize(line)
            lines.append(line)


        return lines, texts



    def convert_input_file_to_tensor_dataset(self,
                                             lines,
                                             args,
                                             tokenizer,
                                             pad_token_label_id,
                                             cls_token_segment_id=0,
                                             pad_token_segment_id=0,
                                             sequence_a_segment_id=0,
                                             mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_slot_label_mask = []


        for words in lines:
            tokens = []
            slot_label_mask = []


            for token in words:
                tokens.append(token)
                if '▁' in token:
                    slot_label_mask.append(0)
                else:
                    slot_label_mask.append(1)


            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > args.max_seq_len - special_tokens_count:
                tokens = tokens[: (args.max_seq_len - special_tokens_count)]
                slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            slot_label_mask += [pad_token_label_id]

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids
            slot_label_mask = [pad_token_label_id] + slot_label_mask

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

            all_input_ids.append(input_ids)
            #print('append', input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
            all_slot_label_mask.append(slot_label_mask)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

        return dataset
    
    def predict(self,input_source):
        # load model and args
        
        
        args = self.get_args()
        device = self.get_device()
        model = self.load_model(args, device)
        label_lst = self.ner_config.label_lst
        logger.info(args)

        # Convert input file to TensorDataset
        pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
        tokenizer = self.tokenizer
        
        self.ner_config.input_source = input_source
        lines,texts = self.read_input_file(tokenizer)
        
        dataset = self.convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)

        # Predict
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.ner_config.batch_size)

        all_slot_label_mask = None
        preds = None



        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3]}
                if args.model_type != "distilkobert":
                    inputs["token_type_ids"] = batch[2]



                outputs = model(**inputs)
                _, logits = outputs[:2]




                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    all_slot_label_mask = batch[3].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(label_lst)}
        preds_list = [[] for _ in range(preds.shape[0])]

        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                if all_slot_label_mask[i, j] != pad_token_label_id:
                    preds_list[i].append(slot_label_map[preds[i][j]])

        # Write to output file
        brands_list = []
        with open(self.ner_config.output_file, "w", encoding="utf-8") as f:
            for words, preds, t in zip(lines, preds_list, texts):
                line = ""
                for word, pred in zip(words, preds):
                    #print(word, pred)
                    if pred == 'O':
                        pass
                    
                    else:
                        if(pred == 'brand-B'):
                            line += '+'
                        line += "{}".format(word)

                if len(line) < 1:
                    line = '__없음__'
                else:
                    l = line.split('+')
                    if len(l) != 1:
                        line = list(set(l))[1]
                brands_list.append(line)
                line = t +', '+ line
                line = re.sub('▁','',line)
                f.write("{}\n".format(line.strip()))


        #logger.info("Prediction Done!")
        return brands_list
    
    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        #
        #print(self.train_dataset[0])
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.ner_config.train_batch_size)


        if self.ner_config.max_steps > 0:
            t_total = self.ner_config.max_steps
            self.ner_config.num_train_epochs = self.ner_config.max_steps // (len(train_dataloader) // self.ner_config.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.ner_config.gradient_accumulation_steps * self.ner_config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.ner_config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.ner_config.learning_rate, eps=self.ner_config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.ner_config.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.ner_config.num_train_epochs)
        logger.info("  Total train batch size = %d", self.ner_config.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.ner_config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.ner_config.logging_steps)
        logger.info("  Save steps = %d", self.ner_config.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.ner_config.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.ner_config.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.ner_config.gradient_accumulation_steps > 1:
                    loss = loss / self.ner_config.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.ner_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ner_config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.ner_config.logging_steps > 0 and global_step % self.ner_config.logging_steps == 0:
                        self.evaluate("test", global_step)

                    if self.ner_config.save_steps > 0 and global_step % self.ner_config.save_steps == 0:
                        self.save_model()

                if 0 < self.ner_config.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.ner_config.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode, step):
        def f1_pre_rec(labels, preds):
            return {
                "precision": precision_score(labels, preds, suffix=True),
                "recall": recall_score(labels, preds, suffix=True),
                "f1": f1_score(labels, preds, suffix=True)
            }
        
        def compute_metrics(labels, preds):
            assert len(preds) == len(labels)
            return f1_pre_rec(labels, preds)

        def show_report(labels, preds):
            return classification_report(labels, preds, suffix=True)
        
        
        print('mode : ', mode)
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        
        
        self.test_texts = [self.tokenizer.convert_ids_to_tokens(d[0][1:]) for d in dataset]
        
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.ner_config.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.ner_config.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.ner_config.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        preds = np.argmax(preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])
                
        if self.ner_config.write_pred:
            if not os.path.exists(self.ner_config.pred_dir):
                os.mkdir(self.ner_config.pred_dir)

            with open(os.path.join(self.ner_config.pred_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    for t, tl, pl in zip(text, true_label, pred_label):
                        f.write("{} {} {}\n".format(t, tl, pl))
                    f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results
    
    
    
    

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.ner_config.model_dir):
            os.makedirs(self.ner_config.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.ner_config.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.ner_config, os.path.join(self.ner_config.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.ner_config.model_dir)
    
    
        





