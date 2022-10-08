import pandas as pd
import numpy as np


from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from sklearn.model_selection import train_test_split
## self-modified
from pipeline_base_modified import PromptDataLoader, PromptForClassification
from template_generation_wzy import LMBFFTemplateGenerationTemplate, T5TemplateGenerator
from openprompt.plms import load_plm
from tqdm import tqdm
import torch.nn as nn
import os
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import ManualTemplate
from openprompt.trainer import ClassificationRunner
import copy
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = True
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")


def sum_list(a,b):
    res = []
    if len(a) >= len(b):
        for i in range(len(b)):
            res.append(a[i]+b[i])# 
        for i in range(len(a) - len(b)):
            res.append(a[len(b)+i])
        return res
    elif len(a) < len(b):
        return sum_list(b, a)


Personalities = ['A','C','E','O','N']


for personality in Personalities:
    with open('label_words/' + personality + '_words.txt', 'r') as f:
        pos = f.readline().split(',')
        neg = f.readline().split(',')

    with open('label_words/' + personality + '_weights.txt', 'r') as f:
        pos_weights = eval(f.readline())
        neg_weights = eval(f.readline())

    for seed in [321, 42, 1024, 0, 1, 13, 41, 123, 456, 999]:
        candidate_templates = []
        with open('templates/Fine_Tuned_Friends_'+personality+'_SEED_'+str(seed)+'_templates_top_10.txt', 'r') as f:
            candidate_templates = [i.strip() for i in f.readlines()]
        
        classes = [0,1]        
        myverbalizer = ManualVerbalizer(
            classes = classes,
            label_words = {
                0 : neg, 
                1 : pos
            },
            tokenizer=tokenizer)

        from openprompt import PromptDataLoader
        batch_size = 1

        overall_logits = []

        for i in range(len(candidate_templates)):
            sample = InputExample(guid=i, text_a="", label=0)
            mytemplate = ManualTemplate(
                text = candidate_templates[i],
                tokenizer = tokenizer,
            )
            prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=True)
            if use_cuda:
                prompt_model = prompt_model.cuda()
            prompt_model.eval()
            validation_dataloader = PromptDataLoader(dataset=[sample], template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=128,
                batch_size=batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                truncate_method="head")

            for step, inputs in enumerate(validation_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model.forward_without_verbalize(inputs)
                label_words_logits = prompt_model.verbalizer.project(logits)
                label_words_probs = prompt_model.verbalizer.normalize(label_words_logits)
                label_words_logits = label_words_probs # torch.log(label_words_probs+1e-15)
                overall_logits.append(label_words_logits.detach())

        overall_logits = torch.cat(overall_logits)
        overall_logits = torch.mean(overall_logits, 0)

        neg_logits = [0.1*float(i) for i in list(overall_logits[0]/overall_logits[0].sum().cpu())]
        pos_logits = [0.1*float(i) for i in list(overall_logits[1]/overall_logits[1].sum().cpu())]

        pos_weights_ = [i/sum(pos_weights) for i in pos_weights]
        neg_weights_ = [i/sum(neg_weights) for i in neg_weights]


        pos_logit_weight = sum_list(pos_weights_, pos_logits)
        neg_logit_weight = sum_list(neg_weights_, neg_logits)



        pos_dict = {}
        for word, weight in zip(pos, pos_logit_weight[:len(pos)]):
            pos_dict[word] = weight

        neg_dict = {}
        for word, weight in zip(neg, neg_logit_weight[:len(neg)]):
            neg_dict[word] = weight  

        # n = 20
        # pos_ = sorted(pos_dict.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)[:n]
        # neg_ = sorted(neg_dict.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)[:n]
        
        # print('Top', n, 'positive label words are:', [i[0] for i in pos_])
        # print('Top', n, 'negative label words are:', [i[0] for i in neg_])

        with open('label_words/posterior_'+personality+'_label_words_SEED_'+str(seed)+'.txt', 'w') as f:
            f.write(','.join(list(pos_dict.keys())))
            f.write(','.join(list(neg_dict.keys())))
        with open('label_words/posterior_'+personality+'_label_weights_SEED_'+str(seed)+'.txt', 'w') as f:
            f.write(str(list(pos_dict.values())))
            f.write('\n')
            f.write(str(list(neg_dict.values())))




                










