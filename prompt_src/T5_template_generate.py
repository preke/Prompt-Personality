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
import torch


class DF_Processor(DataProcessor):
    
    def __init__(self):
        super().__init__()
        self.labels = ['0', '1']

    def get_examples(self, df):
        examples = []
        for i,r in df.iterrows():
            text_a = r['utterance']
            label = r['labels']
            guid = i
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        return examples


from openprompt.prompts import ManualTemplate
from openprompt.trainer import ClassificationRunner
import copy
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup

def fit(model, train_dataloader, val_dataloader, loss_func, optimizer):
    best_score = 0.0
    for epoch in range(10):
        train_epoch(model, train_dataloader, loss_func, optimizer)
        score = evaluate(model, val_dataloader)
        if score > best_score:
            best_score = score
    return best_score

def train_epoch(model, train_dataloader, loss_func, optimizer):
    model.train()
    for step, inputs in enumerate(train_dataloader):
        if cuda:
            inputs = inputs.cuda()
        logits = model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, val_dataloader):
    model.eval()
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            if cuda:
                inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc


def template_post_process(template_texts):
    results = []
    special_tokens = ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']
    for template in template_texts:

        new_tmp = []
        for i in range(len(template)):
            new_tmp.append(template[i])
            try:
                if template[i] in special_tokens and template[i+1] in special_tokens:
                    new_tmp.append('')
            except:
                pass
        results.append(new_tmp)
    return results




import argparse

if __name__ == '__main__':
    
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    for personality in ['A']:#['A', 'C','E','O','N']:
        print('Processing: ', personality)
        
        df_data = pd.read_csv('../data/FriendsPersona/Friends_' + personality + '_whole.tsv', sep='\t')
        df = df_data[['utterance', 'labels']]
        for SEED in [999]:#[42, 1024, 0, 1, 13, 41, 123, 456, 999, 321]:
            print('Seed: ', SEED)
            
            beam_width = 10
            n = 6 # top-n label words for template generation
            
            df_train, df_test, label_train, label_test = \
                        train_test_split(df, df['labels'], test_size=0.1, random_state=SEED, stratify=df['labels'])
            df_train, df_valid, label_train, label_valid = \
                        train_test_split(df_train, label_train, test_size=0.1, random_state=SEED, stratify=label_train)                

            data_train = DF_Processor().get_examples(df_train)
            data_valid = DF_Processor().get_examples(df_valid)

            print('load model...')
            template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = \
                                load_plm('t5', 'Friends_template_t5_base/')
#             template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = \
#                                 load_plm('t5', 't5-base')

            ## how to select label words
            with open('label_words/'+personality+'_words.txt', 'r') as f:
                pos_words = f.readline().split(',')
                neg_words = f.readline().split(',')

            with open('label_words/'+personality+'_weights.txt', 'r') as f:
                pos_weights = eval(f.readline())
                neg_weights = eval(f.readline())

            pos_dict = {}
            for word, weight in zip(pos_words, pos_weights):
                pos_dict[word] = weight

            neg_dict = {}
            for word, weight in zip(neg_words, neg_weights):
                neg_dict[word] = weight

            pos = sorted(pos_dict.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)[:n]
            neg = sorted(neg_dict.items(), key=lambda kv:(kv[1], kv[0]), reverse=False)[:n]
            print('Top', n, 'positive label words are:', pos)
            print('Top', n, 'negative label words are:', neg)

            ## label word with weights? no needs

            classes = [0,1]        
            verbalizer = ManualVerbalizer(
                classes = classes,
                label_words = {
                    0 : [i[0] for i in neg], 
                    1 : [i[0] for i in pos]
                },
                tokenizer=template_generate_tokenizer)

            template = LMBFFTemplateGenerationTemplate(tokenizer = template_generate_tokenizer, 
                                                       verbalizer = verbalizer, 
                                                       text = '{"placeholder":"text_a"} {"mask"} {"meta":"labelword"} {"mask"}.')

            # template generation

            if cuda:
                template_generate_model = template_generate_model.cuda()


            template_generator = T5TemplateGenerator(template_generate_model, 
                                                     template_generate_tokenizer, 
                                                     template_tokenizer_wrapper, 
                                                     verbalizer, 
                                                     beam_width=beam_width) # beam_width is set to 5 here for efficiency, to improve performance, try a larger number.

            dataloader = PromptDataLoader(data_train, 
                                          template, 
                                          tokenizer=template_generate_tokenizer, 
                                          tokenizer_wrapper_class=template_tokenizer_wrapper, 
                                          batch_size=len(data_train), 
                                          decoder_max_length=128, 
                                          max_seq_length=128, 
                                          shuffle=False, 
                                          teacher_forcing=False) # register all data at once

            print('pass!')

            for data in dataloader:
                data = data.cuda()
                print(data.input_ids.shape, data.attention_mask.shape)
                template_generator._register_buffer(data)

            template_texts = template_generator._get_templates()
            _template_texts = template_post_process(template_texts)

            original_template = template.text

            template_texts = []
            for template_text in _template_texts:
                try:
                    tmp = template_generator.convert_template(template_text, original_template)
                    template_texts.append(tmp)
                except:
                    print(template_text)

            # template_generator._show_template()
            template_generator.release_memory()
            # generate a number of candidate template text
            with open('templates/Fine_Tuned_Friends_'+personality+'_SEED_' +str(SEED)+'_templates_top_'+str(beam_width)+'.txt', 'w') as f:
                for tmp in template_texts:
                    f.write(tmp+'\n')























