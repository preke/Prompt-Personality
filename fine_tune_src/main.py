# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

from data_loader import load_data
import shutil

from train import train_model, eval_model
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
# CONFIG

parser = argparse.ArgumentParser(description='')
args   = parser.parse_args()

args.device       = 0 #0, 1
args.MAX_LEN      = 128 # 128, 256 512
args.adam_epsilon = 1e-8
args.epochs       = 3
args.num_class    = 2
args.drop_out     = 0.1
args.test_size    = 0.1


few_shot_list = [-1]
# few_shot_list = [0, 1, 5, 10, 20, -1] # -1 means full
# BASE_list = ['BERT']
BASE_list = ['BERT', 'RoBERTa']




for few_shots in few_shot_list:
    args.few_shot =  few_shots
    
    for base in BASE_list:
        args.BASE = base    
        
    
        if args.BASE == 'BERT':
            args.lr = 1e-5 
        elif args.BASE == 'RoBERTa' or args.BASE == 'RoBERTaLarge':
            args.lr = 1e-4

        args.data         = 'Friends_Persona'
        # args.data         = 'Essay'
#         args.data         = 'MyPersonality'
        # args.data         = 'mbti'
#         args.data         = 'Pan'

        args.result_name  = './src/result/'+args.BASE + '_' + args.data + '_shots_' + str(args.few_shot) + '.txt' 



        if args.BASE == 'BERT':
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        elif args.BASE == 'RoBERTa':
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
        elif args.BASE == 'RoBERTaLarge':
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large", do_lower_case=True)



        cnt = 0

        seeds =  [321, 42, 1024, 0, 1, 13, 41, 123, 456, 999] # 

        if args.data == 'Friends_Persona' or args.data == 'Essay' or args.data == 'MyPersonality' or args.data == 'Pan':
            personalities = ['A', 'C', 'E', 'O', 'N']
            args.batch_size = 8 # 32, 16， 8
        elif args.data == 'mbti':
            personalities = ['I', 'N', 'T', 'J']
            args.batch_size = 8

        with open(args.result_name, 'w') as f:
            test_acc_total = []
            for personality in personalities:
                if args.data == 'Friends_Persona':
                    df = pd.read_csv('./data/Friends_'+personality+'_whole.tsv', sep='\t')
                elif args.data == 'Essay':
                    df = pd.read_csv('./data/Essay_'+personality+'_whole.tsv', sep='\t')
                elif args.data == 'MyPersonality':
                    df = pd.read_csv('./data/MyPersonality_'+personality+'_whole.tsv', sep='\t')
                elif args.data == 'mbti':
                    df = pd.read_csv('./data/mbti_'+personality+'_whole.tsv', sep='\t')
                elif args.data == 'Pan':
                    df = pd.read_csv('./data/Pan_'+personality+'_whole.tsv', sep='\t')
                
                print('Current training classifier for', personality, '...')
                test_acc_all_seeds = []
                for seed in seeds:
                    args.SEED = seed
                    np.random.seed(args.SEED)
                    torch.manual_seed(args.SEED)
                    torch.cuda.manual_seed_all(args.SEED)

                    args.model_path  = './src/model/'  + args.data + '_' + str(args.MAX_LEN) + '_' + args.BASE + '_'+ str(args.lr) +'_' + '_batch_' \
                                        + str(args.batch_size) + '_personality_' + personality + '_seed_' + str(seed) +'_epoch_' + str(args.epochs) + '/'


                    # load data
                    train_dataloader, valid_dataloader, test_dataloader, train_length = load_data(df, args, tokenizer)



                    # load initial model
                    if args.BASE == 'BERT' :
                        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', \
                                num_labels=args.num_class).cuda(args.device)
                    elif args.BASE == 'RoBERTa':
                        model = RobertaForSequenceClassification.from_pretrained('roberta-base', \
                                num_labels=args.num_class).cuda(args.device)
                    elif args.BASE == 'RoBERTaLarge':
                        model = RobertaForSequenceClassification.from_pretrained('roberta-large', \
                                num_labels=args.num_class).cuda(args.device)
                    
                    # train model
                    if args.few_shot != 0: # if zero-shot: no train
                        training_loss, best_eval_acc = train_model(model, args, train_dataloader, valid_dataloader, train_length)

                    # load trained model for test
                    try:
                        if args.BASE == 'BERT' :
                            model = BertForSequenceClassification.from_pretrained(args.model_path, \
                                       num_labels=args.num_class).cuda(args.device)
                        elif args.BASE == 'RoBERTa' or args.BASE == 'RoBERTaLarge':
                            model = RobertaForSequenceClassification.from_pretrained(args.model_path, \
                                        num_labels=args.num_class).cuda(args.device)
                    except:
                        print(traceback.print_exc()) # load the origin model


                    
                    print('Load model from', args.model_path)
                    test_acc = eval_model(model, args, test_dataloader)
                    test_acc_all_seeds.append(test_acc)
                    print('Current Seed is', seed)
                    print('Test F1:', test_acc)
                    print('*'* 10, test_acc_total)
                    print()
                    
                    try:
                        shutil.rmtree(args.model_path)
                    except:
                        print(traceback.print_exc())
                    
                test_acc_total.append(test_acc_all_seeds)
                print('\n========\n')
                print(test_acc_total)
            f.write(str(test_acc_total))



