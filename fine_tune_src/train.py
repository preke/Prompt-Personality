import sklearn
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import tqdm, trange,tnrange, tqdm_notebook
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import traceback
import os
import shutil



def train_model(model, args, train_dataloader, valid_dataloader, train_length):
    if args.data == 'Friends_Persona':
        num_warmup_steps = int(0*train_length)
    else:
        num_warmup_steps   = int(0.05*train_length)

    num_training_steps = len(train_dataloader)*args.epochs
    

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    
    train_loss_set = []
    learning_rate  = []
    model.zero_grad()
    best_eval_acc = 0

    for _ in tnrange(1, args.epochs+1, desc='Epoch'):
        print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
        # Calculate total loss for this epoch
        batch_loss = 0

        for step, batch in enumerate(train_dataloader):
            # Set our model to training mode (as opposed to evaluation mode)
            model.train()
            
            # Add batch to GPU
            batch = tuple(t.cuda(args.device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss    = outputs.loss
            logits  = outputs.logits
            
            # Backward pass
            loss.backward()
            
            # Clip the norm of the gradients to 1.0
            # Gradient clipping is not in AdamW anymore
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update tracking variables
            batch_loss += loss.item()


            # Validation
            # Put model in evaluation mode to evaluate loss on the validation set
            if step%5 == 0:
                eval_acc = eval_model(model, args, valid_dataloader)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    try:
                        shutil.rmtree(args.model_path)
                    except:
                        print(traceback.print_exc())
                        os.mkdir(args.model_path)
                    try:
                        model.save_pretrained(args.model_path)
                        print('****** saved new model to ' + args.model_path + ' ******')
                    except:
                        print(traceback.print_exc())
                else:
                    print('EVAL F1:', eval_acc, ' ', 'BEST F1', best_eval_acc)

        # Calculate the average loss over the training data.
        avg_train_loss = batch_loss / len(train_dataloader)

          
        train_loss_set.append(avg_train_loss)
        print(F'\n\tAverage Training loss: {avg_train_loss}')
          
    return train_loss_set, best_eval_acc

    
    
    

def eval_model(model, args, valid_dataloader):
    # Tracking variables 
    eval_accuracy, eval_mcc_accuracy, nb_eval_steps = 0, 0, 0
    
    labels_list = np.array([])
    pred_list = np.array([])

    # Evaluate data for one epoch
    for batch in valid_dataloader:
        # Add batch to GPU
        batch = tuple(t.cuda(args.device) for t in batch)
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        model.eval()
        with torch.no_grad():
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            logits  = outputs.logits

        # Move logits and labels to CPU
        logits      = logits.to('cpu').numpy()
        label_ids   = b_labels.to('cpu').numpy()
        
        pred_flat   = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        
        pred_list   = np.append(pred_list, pred_flat)
        labels_list = np.append(labels_list, labels_flat)
                
        nb_eval_steps += 1

    
    return f1_score(labels_list, pred_list)


