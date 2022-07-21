import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,IterableDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import json
import re



def few_shot_sampler(uttrs, labels, uttr_masks, few_shot):
    pos                 = 0
    neg                 = 0
    few_shot_uttrs      = []
    few_shot_labels     = []
    few_shot_uttr_masks = []
    
    for i in range(len(labels)):
        if labels[i] == 1 and pos < few_shot:
            few_shot_uttrs.append(uttrs[i])
            few_shot_labels.append(labels[i])
            few_shot_uttr_masks.append(uttr_masks[i])
            pos += 1
        elif labels[i] == 0 and neg < few_shot:
            few_shot_uttrs.append(uttrs[i])
            few_shot_labels.append(labels[i])
            few_shot_uttr_masks.append(uttr_masks[i])
            neg += 1

        if pos >= few_shot and neg >= few_shot:
            break
    return few_shot_uttrs, few_shot_labels, few_shot_uttr_masks


def load_data(df, args, tokenizer):
    uttrs = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN, \
            pad_to_max_length=True) for sent in df['utterance']]
    uttr_masks = [[float(i>0) for i in seq] for seq in uttrs]
    
    labels = list(df['labels'])
    
    # train dev test split
    train_uttrs, test_uttrs, train_labels, test_labels = \
        train_test_split(uttrs, labels, random_state=args.SEED, test_size=args.test_size, stratify=labels)
    train_uttr_masks, test_uttr_masks,_,_ = train_test_split(uttr_masks,labels,random_state=args.SEED, \
                                                   test_size=args.test_size,  stratify=labels)
    
    train_set_labels = train_labels
    
    train_uttrs, valid_uttrs, train_labels, valid_labels = \
        train_test_split(train_uttrs, train_set_labels, random_state=args.SEED, test_size=args.test_size, stratify=train_set_labels)
    train_uttr_masks, valid_uttr_masks,_,_ = train_test_split(train_uttr_masks, train_set_labels, random_state=args.SEED, \
                                                   test_size=args.test_size,  stratify=train_set_labels)
    
    '''
    For a k-shot experiment, we sample k instances of each class from the original training set 
    to form the few- shot training set and sample another k instances per class to form the validation set. 
    '''

    if args.few_shot > 0:
        train_uttrs, train_labels, train_uttr_masks = few_shot_sampler(train_uttrs, train_labels, train_uttr_masks, args.few_shot)
        valid_uttrs, valid_labels, valid_uttr_masks = few_shot_sampler(valid_uttrs, valid_labels, valid_uttr_masks, args.few_shot)
        print('Length of train set', len(train_uttrs))
        print('Length of valid set', len(valid_uttrs))
    else:
        pass
        

    train_uttrs         = torch.tensor(train_uttrs)
    valid_uttrs         = torch.tensor(valid_uttrs)
    test_uttrs          = torch.tensor(test_uttrs)

    train_uttr_masks    = torch.tensor(train_uttr_masks)
    valid_uttr_masks    = torch.tensor(valid_uttr_masks)
    test_uttr_masks     = torch.tensor(test_uttr_masks)
    
    train_labels        = torch.tensor(train_labels)    
    valid_labels        = torch.tensor(valid_labels)
    test_labels         = torch.tensor(test_labels)

    train_data       = TensorDataset(train_uttrs, train_uttr_masks, train_labels)
    train_sampler    = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    valid_data       = TensorDataset(valid_uttrs, valid_uttr_masks, valid_labels)
    valid_sampler    = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
    
    test_data        = TensorDataset(test_uttrs, test_uttr_masks, test_labels)
    test_sampler     = RandomSampler(test_data)
    test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
    
    train_length     = len(train_data)
    return train_dataloader, valid_dataloader, test_dataloader, train_length
   