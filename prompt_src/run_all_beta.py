import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import  AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import argparse

from openprompt import PromptDataLoader
from openprompt.plms import load_plm
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt import PromptForClassification

from openprompt.prompts import ManualVerbalizer, ManualTemplate


import json
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
import os
from openprompt.prompts.manual_template import ManualTemplate
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompts import ManualTemplate
from openprompt.prompts.prompt_generator import T5TemplateGenerator
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.trainer import ClassificationRunner
import copy
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()
import os
args.device         = torch.device('cuda:1') 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
args.adam_epsilon   = 1e-8
args.num_of_epoches = 3
args.num_class      = 2
args.drop_out       = 0.1
args.test_size      = 0.1
args.method         = 'LM-BFF' #'KPT' #'PET_wiki', 'KPT_augmented'

# knowledgeable prompt tuning:

args.candidate_frac  = 0.5
args.pred_temp       = 1.0
args.max_token_split = -1



# datasets      = ['MyPersonality', 'Pan', 'mbti', 'Friends_Persona', 'Essay']
datasets      = ['Friends_Persona']# ['Friends_Persona', 'Essay', 'MyPersonality', 'Pan']
seeds         = [42]# [321, 42, 1024, 0, 1, 13, 41, 123, 456, 999]
# few_shot_list = [0, 1, 5, 10, 20, -1]
few_shot_list = [0]# [0, 1, 5, 10, 20, -1]
BASE_list     = ['RoBERTa']# , 'RoBERTa-large'] # BERT




MAX_LEN_dict = {
    'Friends_Persona': 128,
    'Essay'          : 512,
    'MyPersonality'  : 512,
    'Pan'            : 512,
    # 'mbti'           : 512
}

Personalities_dict = {
    'Friends_Persona': ['A'], #['A', 'C', 'E', 'O', 'N'],
    'Essay'          : ['A', 'C', 'E', 'O', 'N'],
    'MyPersonality'  : ['A', 'C', 'E', 'O', 'N'],
    'Pan'            : ['A', 'C', 'E', 'O', 'N'],
    # 'mbti'           : ['I', 'N', 'T', 'J']
}


DATA_PATH_dict = {
    'Friends_Persona': '../data/FriendsPersona/Friends_',
    'Essay'          : '../data/Essay/Essay_',
    'MyPersonality'  : '../data/myPersonality/MyPersonality_',
    'Pan'            : '../data/pan2015/Pan_',
    # 'mbti'           : '../data/Kaggle_mbti/mbti_'

}



default_labelwords = {
    'A': [['friendly', 'compassionate'], ['critical', 'rational']],
    'C': [['efficient', 'organized'], ['extravagant', 'careless']],
    'E': [['outgoing', 'energetic'], ['solitary', 'reserved']],
    'O': [['inventive', 'curious'], ['consistent', 'cautious']],
    'N': [['sensitive', 'nervous'], ['resilient', 'confident']]
}

cuda = True

# ==============

def evaluation(validation_dataloader, prompt_model):
    prompt_model.eval()
    labels_list = np.array([])
    pred_list = np.array([])
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        preds  = torch.argmax(logits, dim=-1).cpu().tolist()
        labels_list = np.append(labels.cpu().tolist(), labels_list)
        pred_list = np.append(preds, pred_list)
    return f1_score(labels_list, pred_list)


def calibrate(prompt_model: PromptForClassification, dataloader: PromptDataLoader) -> torch.Tensor:
    r"""Calibrate. See `Paper <https://arxiv.org/abs/2108.02035>`_
    
    Args:
        prompt_model (:obj:`PromptForClassification`): the PromptForClassification model.
        dataloader (:obj:`List`): the dataloader to conduct the calibrate, could be a virtual one, i.e. contain an only-template example.
    
    Return:
        (:obj:`torch.Tensor`) A tensor of shape  (vocabsize) or (mask_num, vocabsize), the logits calculated for each word in the vocabulary
    """
    all_logits = []
    prompt_model.eval()
    for batch in tqdm(dataloader,desc='ContextCali'):
        batch = batch.cuda()
        logits = prompt_model.forward_without_verbalize(batch)
        all_logits.append(logits.detach())
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits

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
    from sklearn.metrics import f1_score
#     acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    f1 = f1_score(alllabels, allpreds)
    return f1        


def lmbff_template_generation(dataset, verbalizer):
    default_labelwords
    # load generation model for template generation
    template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm('t5', 't5-base')

    from openprompt.prompts import ManualTemplate

    from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate
    template = LMBFFTemplateGenerationTemplate(tokenizer=template_generate_tokenizer, verbalizer=verbalizer, text='{"placeholder":"text_a"} {"mask"} {"meta":"labelword"} {"mask"}.')
    print('performing auto_t...')
    # load generation model for template generation
    template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm('t5', 't5-base')


    if cuda:
        template_generate_model = template_generate_model.cuda()
    template_generator = T5TemplateGenerator(template_generate_model, template_generate_tokenizer, template_tokenizer_wrapper, verbalizer, beam_width=1) # beam_width is set to 5 here for efficiency, to improve performance, try a larger number.

    dataloader = PromptDataLoader(dataset['train'], template, tokenizer=template_generate_tokenizer, tokenizer_wrapper_class=template_tokenizer_wrapper, batch_size=len(dataset['train']), decoder_max_length=128, max_seq_length=128, shuffle=False, teacher_forcing=False) # register all data at once
    
    for data in dataloader:
        if cuda:
            data = data.cuda()
        template_generator._register_buffer(data)

    template_generate_model.eval()
    print('generating...')
    template_texts = template_generator._get_templates()

    original_template = template.text
    template_texts = [template_generator.convert_template(template_text, original_template) for template_text in template_texts]
#     template_generator._show_template()
    template_generator.release_memory()
    # generate a number of candidate template text
    print(template_texts)
    # iterate over each candidate and select the best one
    best_metrics = 0.0
    best_template_text = None
    for template_text in tqdm(template_texts):
        template = ManualTemplate(tokenizer, template_text)

        train_dataloader = PromptDataLoader(dataset['train'], template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass)
        valid_dataloader = PromptDataLoader(dataset['validation'],template,tokenizer=tokenizer,tokenizer_wrapper_class=WrapperClass)

        model = PromptForClassification(copy.deepcopy(plm), template, verbalizer)

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        if cuda:
            model = model.cuda()
        score = fit(model, train_dataloader, valid_dataloader, loss_func, optimizer)
        print('*'*20)
        print(score)
        print('*'*20)
        if score > best_metrics:
            print('best score:', score)
            print('template:', template_text)
            best_metrics = score
            best_template_text = template_text
    # use the best template
    template = ManualTemplate(tokenizer, text=best_template_text)
    print(best_template_text)
    return template

def lmbff_verbalizer_generation(dataset, template):
    from openprompt.prompts.prompt_generator import RobertaVerbalizerGenerator
    # load generation model for template generation
    # load mlm model for main tasks
    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")
    
    from openprompt.prompts import ManualVerbalizer, ManualTemplate
    class_labels = [0, 1]
    verbalizer = ManualVerbalizer(
                                classes = class_labels,
                                label_words = {
                                    0 : default_labelwords[args.personality][1], 
                                    1 : default_labelwords[args.personality][0]
                                },
                                tokenizer=tokenizer)
    if cuda:
        plm = plm.cuda()
    verbalizer_generator = RobertaVerbalizerGenerator(model=plm, tokenizer=tokenizer, candidate_num=2, label_word_num_per_class=10)
    # to improve performance , try larger numbers

    dataloader = PromptDataLoader(dataset['train'], template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, batch_size=32)
    for data in dataloader:
        if cuda:
            data = data.cuda()
        verbalizer_generator.register_buffer(data)
    
    label_words_list = verbalizer_generator.generate()
    verbalizer_generator.release_memory()

    # iterate over each candidate and select the best one
    current_verbalizer = copy.deepcopy(verbalizer)
    best_metrics = 0.0
    best_label_words = None
    for label_words in tqdm(label_words_list):
        current_verbalizer.label_words = label_words
        print(current_verbalizer.label_words)
        
        train_dataloader = PromptDataLoader(dataset['train'], template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass)
        valid_dataloader = PromptDataLoader(dataset['validation'], template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass)

        model = PromptForClassification(copy.deepcopy(plm), template, current_verbalizer)

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        if cuda:
            model = model.cuda()
        score = fit(model, train_dataloader, valid_dataloader, loss_func, optimizer)

        if score > best_metrics:
            best_metrics = score
            best_label_words = label_words
    # use the best verbalizer
    verbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=best_label_words)
    return verbalizer



# ==============


def data_loader(args, dataset, mytemplate, tokenizer, WrapperClass):
    
    if args.shots > 0:
        args.batch_size = 1
    else:
        args.batch_size = 8

    train_dataloader = PromptDataLoader(
        dataset                 = dataset["train"], 
        template                = mytemplate, 
        tokenizer               = tokenizer,
        tokenizer_wrapper_class = WrapperClass, 
        max_seq_length          = MAX_LEN, 
        batch_size              = args.batch_size, 
        shuffle                 = True, 
        teacher_forcing         = False, 
        predict_eos_token       = False,
        truncate_method         = "head")

    validation_dataloader = PromptDataLoader(
        dataset                 = dataset["validation"], 
        template                = mytemplate, 
        tokenizer               = tokenizer,
        tokenizer_wrapper_class = WrapperClass, 
        max_seq_length          = MAX_LEN, 
        batch_size              = args.batch_size, 
        shuffle                 = True, 
        teacher_forcing         = False, 
        predict_eos_token       = False,
        truncate_method         = "head")

    test_dataloader = PromptDataLoader(
        dataset                 = dataset["test"], 
        template                = mytemplate, 
        tokenizer               = tokenizer,
        tokenizer_wrapper_class = WrapperClass, 
        max_seq_length          = MAX_LEN, 
        batch_size              = args.batch_size, 
        shuffle                 = True, 
        teacher_forcing         = False, 
        predict_eos_token       = False,
        truncate_method         = "head")

    return train_dataloader, validation_dataloader, test_dataloader



# ================ dataset selection ================
for data in datasets:
    args.data = data
    
    personalities = Personalities_dict[args.data]
    MAX_LEN       = MAX_LEN_dict[args.data]
    
    # ================ few shot number selection ================
    for shots in few_shot_list:
        args.shots = shots
        # ================ pre-trained model selection ================
        for pre_trained_model in BASE_list:
            args.BASE = pre_trained_model

            # ******* learning rate *******
            if args.BASE == 'BERT':
                args.learning_rate = 1e-5
            elif args.BASE == 'RoBERTa':
                args.learning_rate = 1e-4
            elif args.BASE == 'RoBERTa-large':
                args.learning_rate = 1e-4 

            # ******** result name ********
            args.result_name  = './result/'+args.method+'_Prompt_'+args.BASE + '_' + args.data + '_shots_' + str(args.shots) + '.txt'
            with open(args.result_name, 'w') as f:
                test_f1_total = []
                # ================ personality label selection ================
                for personality in personalities:
                    args.personality = personality
                    
                    df_data = pd.read_csv( DATA_PATH_dict[args.data] + args.personality + '_whole.tsv', sep = '\t')
                    df = df_data[['utterance', 'labels']]
                    
                    
                    test_f1_all_seeds = []
                    # ================ random seed selection ================
                    for SEED in seeds:
                        args.SEED = SEED

                        np.random.seed(args.SEED)
                        torch.manual_seed(args.SEED)
                        torch.cuda.manual_seed_all(args.SEED)
                        torch.backends.cudnn.deterministic = True
                        os.environ["PYTHONHASHSEED"] = str(args.SEED)
                        torch.backends.cudnn.benchmark = False
                        torch.set_num_threads(1)

                        # ******** load pre_trained model ********
            
                        ### need reload for every random seed
                        if args.BASE == 'BERT':
                            plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
                        elif args.BASE == 'RoBERTa':
                            plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")
                        elif args.BASE == 'RoBERTa-large':
                            plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-large")
                        

                        # ******** train test split ********
                        Uttr_train, Uttr_test, label_train, label_test = \
                            train_test_split(df['utterance'], df['labels'], test_size=0.1, random_state=SEED, stratify=df['labels'])

                        Uttr_train, Uttr_valid, label_train, label_valid = \
                            train_test_split(Uttr_train, label_train, test_size=0.1, random_state=SEED, stratify=label_train)

                        # ******** construct samples ********
                        dataset = {}
                        for split in ['train', 'validation', 'test']:
                            dataset[split] = []
                            cnt = 0
                            if split == 'train':
                                for u,l in zip(Uttr_train, label_train):
                                    input_sample = InputExample(text_a=u, label=int(l),guid=cnt)
                                    cnt += 1
                                    dataset[split].append(input_sample)
                            elif split == 'validation':
                                for u,l in zip(Uttr_valid, label_valid):
                                    input_sample = InputExample(text_a=u, label=int(l),guid=cnt)
                                    cnt += 1
                                    dataset[split].append(input_sample)
                            elif split == 'test':
                                for u,l in zip(Uttr_test, label_test):
                                    input_sample = InputExample(text_a=u, label=int(l),guid=cnt)
                                    cnt += 1
                                    dataset[split].append(input_sample)
                

                        # ******** few shot setting ********
                        if args.shots > 0:
                            sampler = FewShotSampler(num_examples_per_label=args.shots, also_sample_dev=True, num_examples_per_label_dev=args.shots)
                            dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.SEED)
                        elif args.shots == 0:
                            # no train
                            pass
                        else:
                            pass

                        # ***** template setting *****

                        #############
                        #############
                        #############
                        class_labels = [0,1]
                        # Manual Template 
                        if args.method == 'LM-BFF':
                            tmp_verbalizer = ManualVerbalizer(
                                classes = class_labels,
                                label_words = {
                                    0 : default_labelwords[args.personality][1], 
                                    1 : default_labelwords[args.personality][0]
                                },
                                tokenizer=tokenizer)
                            mytemplate = lmbff_template_generation(dataset, tmp_verbalizer)
                        else:
                            mytemplate = ManualTemplate(
                                    text = '{"placeholder":"text_a"} I am a {"mask"} person',
                                    tokenizer = tokenizer,
                            )
                        
                        wrapped_tokenizer = WrapperClass(max_seq_length=MAX_LEN, tokenizer=tokenizer,truncate_method="head")

                        model_inputs = {}
                        for split in ['train', 'validation', 'test']:
                            model_inputs[split] = []
                            for sample in dataset[split]:
                                tokenized_example = wrapped_tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
                                model_inputs[split].append(tokenized_example)

                        # ***** data loader ***** 
                        
                        train_dataloader, validation_dataloader, test_dataloader = data_loader(args, dataset, mytemplate, tokenizer, WrapperClass)
                        
                        # ***** construct verbalizer ***** 
                        
                        #############
                        #############
                        #############

                        
                        
                        # vanilla verbalizer
                        if args.method == 'PET_wiki':
                            with open('big_five_'+args.personality+'.txt', 'r') as f_verbalizer:
                                pos = [i.strip() for i in f_verbalizer.readline().split(',')]
                                neg = [i.strip() for i in f_verbalizer.readline().split(',')]
                            
                            myverbalizer = ManualVerbalizer(
                                                classes = class_labels,
                                                label_words = {
                                                    0 : default_labelwords[args.personality][1], 
                                                    1 : default_labelwords[args.personality][0]
                                                },
                                                tokenizer=tokenizer)
                        elif args.method == 'KPT':
                        # Knowledgeable verbalizer
                            myverbalizer = KnowledgeableVerbalizer(
                                    tokenizer, 
                                    classes         = class_labels, 
                                    candidate_frac  = args.candidate_frac, 
                                    pred_temp       = args.pred_temp, 
                                    max_token_split = args.max_token_split).from_file('kpt_label_words/' + args.personality + '_words.txt')
                        elif args.method == 'KPT_augmented':
                            myverbalizer = KnowledgeableVerbalizer(
                                    tokenizer, 
                                    classes         = class_labels, 
                                    candidate_frac  = args.candidate_frac, 
                                    pred_temp       = args.pred_temp, 
                                    max_token_split = args.max_token_split).from_file('label_words/' + args.personality + '_words.txt')
                        elif args.method == 'LM-BFF':
                            myverbalizer = lmbff_verbalizer_generation(dataset, mytemplate)

                        # ***** training setting *****
                        use_cuda = True
                        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
                        if use_cuda:
                            prompt_model =  prompt_model.cuda()

                        loss_func = torch.nn.CrossEntropyLoss()
                        no_decay = ['bias', 'LayerNorm.weight']
                        # it's always good practice to set no decay to biase and LayerNorm parameters
                        
                        # for name, param in prompt_model.named_parameters(): 
                        #     print(name)
                        
                        optimizer_grouped_parameters = [
                            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                        ]

                        best_eval = 0
                        best_test = 0
                        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
                        

                        # ***** training process *****
                        if args.shots != 0:
                            prompt_model.train()
                            for epoch in range(args.num_of_epoches):
                                tot_loss = 0
                                for step, inputs in enumerate(train_dataloader):
                                    if use_cuda:
                                        inputs = inputs.cuda()
                                    
                                    logits = prompt_model(inputs)
                                    labels = inputs['label']
                                    loss   = loss_func(logits, labels)
                                    
                                    loss.backward()
                                    tot_loss += loss.item()
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    if step % 5 == 0: # if step % 100 == 1:
                                        eval_f1 = evaluation(validation_dataloader, prompt_model)
                                        if eval_f1 > best_eval:
                                            best_eval = eval_f1
                                            best_test = evaluation(test_dataloader, prompt_model)
                                        prompt_model.train()
                        else:
                            if args.method.startswith('KPT'):
                                # need calibration
                                myrecord = ""
                                from openprompt.data_utils.data_sampler import FewShotSampler
                                support_sampler = FewShotSampler(num_examples_total=100, also_sample_dev=False)
                                dataset['support'] = support_sampler(dataset['train'], seed=args.SEED)

                                # for example in dataset['support']:
                                #     example.label = -1 # remove the labels of support set for clarification
                                support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3,
                                    batch_size=8,shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")

                                org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
                                # calculate the calibration logits
                                cc_logits = calibrate(prompt_model, support_dataloader)
                                print("the calibration logits is", cc_logits)
                                myrecord += "Phase 1 {}\n".format(org_label_words_num)

                                myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
                                new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
                                myrecord += "Phase 2 {}\n".format(new_label_words_num)
                            else:
                                pass
                            best_test = evaluation(test_dataloader, prompt_model)
                        
                        print('Current SEED:', args.SEED, 'TEST F1', best_test)
                        test_f1_all_seeds.append(best_test)

                    test_f1_total.append(test_f1_all_seeds)
                    print('\n========\n')
                    print(test_f1_total)
                f.write(str(test_f1_total))





