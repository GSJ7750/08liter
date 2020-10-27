import os
import logging
import argparse
from tqdm import tqdm, trange
import regex as re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification
from data_loader import load_and_cache_examples
from utils import init_logger, load_tokenizer, get_labels

logger = logging.getLogger(__name__)



def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model

def clean_reivew(review):
    review = ''.join(re.compile('[가-힣.!?a-zA-Z ]').findall(review))
    review = re.sub('\n', ' ', review)
    review = re.sub(r'\!+', '!', review)
    review = re.sub(r'\?+', '?', review)
    review = re.sub(r'\.+', '.', review)
    review = re.sub(r'\([^)]*\)', '', review)    # 괄호 안 내용 삭제
    review = re.sub(r'\[[^)]*\]', '', review)    # 대괄호 안 내용 삭제
    return review 

def read_input_file(pred_config, tokenizer):
    args = get_args(pred_config)
    
    lines = []
    texts = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            line = clean_reivew(line)
            
            texts.append(line)
            
            line = tokenizer.tokenize(line)
            lines.append(line)
            
            
            #words = line.split()
            #lines.append(words)
            
            


    return lines, texts



def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
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

    
    #print(lines)
                
    
    for words in lines:
        tokens = []
        slot_label_mask = []
        
        
        for token in words:
            tokens.append(token)
            if '▁' in token:
                slot_label_mask.append(0)
            else:
                slot_label_mask.append(1)
                
        #print(words, tokens, slot_label_mask)     
        #assert False
        
        
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
        #print(tokens, input_ids)

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


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    label_lst = get_labels(args)
    logger.info(args)

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    lines,texts = read_input_file(pred_config, tokenizer)
    #print([l for l in lines[:5]])
    

    
    
    
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)
    
    
    
    print(tokenizer.convert_ids_to_tokens(dataset[2][0]),'\n')
    print(dataset[2][1],'\n')
    print(dataset[2][2],'\n')
    print(dataset[2][3],'\n')
    
    
    # Predict
    sampler = SequentialSampler(dataset)
    #print(dataset[0:5])
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

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
                
                
            #for i in range(10):
            #    print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
            #    print(inputs['input_ids'][i])
            #    print(inputs['attention_mask'][i])
            #
            #assert False    
                
            
            outputs = model(**inputs)
            _, logits = outputs[:2]
            #print(len(logits[0]))

            
            
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
                #print(len(preds[0]), len(all_slot_label_mask[0]))
                #assert False
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    print('pred', preds[2], len(preds[2]))
    
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])
                
    print('pred list', preds_list[2], len(preds_list[2]))
    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, preds, t in zip(lines, preds_list, texts):

            
            
            line = ""
            for word, pred in zip(words, preds):
                #print(word, pred)
                if pred == 'O':
                    #line = line + word + ""
                    
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
            line = t +', '+ line
            line = re.sub('▁','',line)
            f.write("{}\n".format(line.strip()))

            
    logger.info("Prediction Done!")



