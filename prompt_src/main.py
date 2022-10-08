import pandas as pd
import numpy as np
from verbalizer import ManualVerbalizer, KnowledgeableVerbalizer
from transformers import  AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report


def evaluation(validation_dataloader):
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




import argparse
from openprompt import PromptDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='roberta')
args = parser.parse_args(args=[])

args.model = 'roberta'
args.model_name_or_path = 'roberta-large'
args.pred_temp = 1.0
args.max_token_split = -1




for personality in ['A', 'C', 'E', 'O', 'N']:
    df_data = pd.read_csv('../data/FriendsPersona/Friends_'+personality+'_whole.tsv', sep = '\t')
    df = df_data[['utterance', 'labels']]
    df_verbalizer = pd.read_csv('big_five_cleaned.tsv', sep='\t')
    pos = [a.lower() for a in list(df_verbalizer['word'][df_verbalizer[personality]>0])]
    neg = [a.lower() for a in list(df_verbalizer['word'][df_verbalizer[personality]<0])]

    seeds =  [321, 42, 1024, 0, 1, 13, 41, 123, 456, 999]
    best_f1s = []
    for SEED in seeds:
        
        # Train test split:
        from sklearn.model_selection import train_test_split
        Uttr_train, Uttr_test, label_train, label_test = \
            train_test_split(df['utterance'], df['labels'], test_size=0.1, random_state=SEED, stratify=df['labels'])

        Uttr_train, Uttr_valid, label_train, label_valid = \
            train_test_split(Uttr_train, label_train, test_size=0.1, random_state=SEED, stratify=label_train)


        # Construct Samples
        from openprompt.data_utils import InputExample
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
        
        
        SHOT = 10
        from openprompt.data_utils.data_sampler import FewShotSampler
        sampler = FewShotSampler(num_examples_per_label=SHOT, also_sample_dev=True, num_examples_per_label_dev=SHOT)
        dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=SEED)

        # Base Model
        from openprompt.plms import load_plm
        # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
        plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")
        # plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-large")

        ## Construct Template and Wrapping
        '''
        #############
        #############
        #############
        '''
        
        # ***** Manual Template *****

        from openprompt.prompts import ManualTemplate
        mytemplate = ManualTemplate(
            text = '{"placeholder":"text_a"} I am a {"mask"} person',
            tokenizer = tokenizer,
        )
        
        # ***** Soft(Mixed) Templeate *****
        
        '''
        from openprompt.prompts import MixedTemplate
        mytemplate = MixedTemplate(
            model = plm, 
            tokenizer = tokenizer,
            text='{"placeholder":"text_a"} {"soft": "He is"} {"mask"} {"soft": "person"}.')
        '''
        
        wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
        wrapped_tokenizer = WrapperClass(max_seq_length=128, tokenizer=tokenizer,truncate_method="head")


        model_inputs = {}
        for split in ['train', 'validation', 'test']:
            model_inputs[split] = []
            for sample in dataset[split]:
                tokenized_example = wrapped_tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
                model_inputs[split].append(tokenized_example)


        ## Data Loader
        from openprompt import PromptDataLoader
        batch_size = 1
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=128, 
            batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")

        validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=128,
            batch_size=batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")

        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=128,
            batch_size=batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="head")


        ## Construct Verbalizer
        '''
        #############
        #############
        #############
        '''
        # from openprompt.prompts import ManualVerbalizer
        import torch

        class_labels = [0,1]
        
        ## vanilla verbalizer
        # myverbalizer = ManualVerbalizer(
        #                     classes = class_labels,
        #                     label_words = {
        #                         0 : neg, 
        #                         1 : pos
        #                     },
        #                     tokenizer=tokenizer)

        ## Knowledgeable verbalizer
        myverbalizer = KnowledgeableVerbalizer(
                tokenizer, 
                classes=class_labels, 
                candidate_frac=0.5, 
                pred_temp=args.pred_temp, 
                max_token_split=args.max_token_split).from_file('big_five_'+personality+'.txt')
        
        
        logits = torch.randn(2, len(tokenizer)) # creating a pseudo output from the plm, and

        from openprompt import PromptForClassification
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


        ## Training
        '''
        #############
        #############
        #############
        '''


        best_eval = 0
        best_test = 0
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        prompt_model.train()
        for epoch in range(3):
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step % 5 == 0: # if step % 100 == 1:
                    eval_f1 = evaluation(validation_dataloader)
                    if eval_f1 > best_eval:
                        best_eval = eval_f1
                        best_test = evaluation(test_dataloader)
                    prompt_model.train()
        print('Current SEED:', SEED, 'TEST F1', best_test)
        best_f1s.append(best_test)

    print(round(np.mean(best_f1s), 3), '$\pm$', round(np.std(best_f1s), 3))
