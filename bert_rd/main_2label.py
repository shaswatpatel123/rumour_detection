import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import preprocessor as p
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from pathlib import PurePath
import pickle
import tabulate
import sys
# from config import *

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# +
MAX_LENGTH = 512
DATAPATH = sys.argv[1] # "./unaugmented"
PRETRAINED_MODEL = sys.argv[2] # "bert-base-uncased"
SAVEPATH = sys.argv[3]
LR = 1e-5
WEIGHT_DECAY = 1e-4
batch_size = 16
EPOCH = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(DATAPATH)
print(PRETRAINED_MODEL)
# -

os.makedirs( os.path.join(SAVEPATH), exist_ok=True )


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

        self.labels = df['label'].to_list()
        self.texts = [self.tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

def preprocess_tweets(data):
    return data.apply(lambda row: p.clean(row))


def checkEvent(event):
    event = PurePath(event).parts[-1]
    if event in ["charliehebdo", "sydneysiege", "ottawashooting", "germanwings", "ferguson"]:
        return True
    return False


directories = [ os.path.join(DATAPATH, o) for o in os.listdir(DATAPATH) if os.path.isdir( os.path.join(DATAPATH, o) ) ]
directories = [ i for i in directories if checkEvent(i) ]
kf = KFold(n_splits=len(directories))

def getLabel( label ):
    if label == "rumours":
        return 1
    return 0

def get_train_test(directories, train, test):
    train_list, train_label_list, test_list, test_label_list = [], [], [], []
    for i in train:
        print( directories[i] )
        tr_path = os.path.join(directories[i],  f"graph.pickle")
        with open(tr_path, "rb") as handle:
            data = pickle.load(handle)

        for d in data:
            train_list.append(d["x"])
            train_label_list.append(getLabel(d["label"]))

    for i in test:
        tst_path = os.path.join(directories[i],  f"graph_test.pickle")
        with open(tst_path, "rb") as handle:
            data = pickle.load(handle)

        for d in data:
            test_list.append(d["x"])
            test_label_list.append(getLabel(d["label"]))

    train_csv = pd.DataFrame({
        "text": train_list,
        "label": train_label_list
    })

    test_csv = pd.DataFrame({
        "text": test_list,
        "label": test_label_list
    })

    return train_csv, test_csv


def printTable(metric_score):
    l = [ [i, j] for i, j in metric_score.items() ]

    table = tabulate.tabulate(l, headers=['Metric', 'Value'], tablefmt='orgtbl')
    print(table)

# +
acc_set = list()
prec_set = list()
recall_set = list()
f1_set = list()
 
for train, test in kf.split(directories):
    print("*" * 60)
    print("Testing: ", PurePath(directories[test[0]]).parts[-1])

    df_train, df_test = get_train_test(directories, train, test)
    
    print( df_train.groupby('label').count() )
    print( df_test.groupby('label').count() )
    
    train_dataset, val_dataset = Dataset(df_train), Dataset(df_test)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    model = AutoModelForSequenceClassification.from_pretrained( PRETRAINED_MODEL, num_labels=2 )
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    
    progressBar = tqdm( range( EPOCH ) )
    for eid in progressBar:
        
        progressText = ""
        
        total_loss_train, total_loss_val = 0,  0
        
        y_true, y_pred = [], []
        train_progressBar = tqdm(train_dataloader)
        for train_input, train_label in train_progressBar:
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            output = model( input_ids=input_id, attention_mask=mask).logits

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            y_pred.extend( output.argmax(dim=1).cpu().tolist() )
            y_true.extend( train_label.cpu().tolist() )

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
        progressText = progressText + f"Train acc: {accuracy_score( y_true, y_pred, normalize=True ):.4f}"
            
        y_true, y_pred = [], []
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_ids=input_id, attention_mask=mask).logits

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                y_pred.extend( output.argmax(dim=1).cpu().tolist() )
                y_true.extend( val_label.cpu().tolist() )
                
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support( y_true, y_pred, average='macro') # micro
        val_acc = accuracy_score( y_true, y_pred, normalize=True )
        progressText = progressText + f", Test acc: {val_acc:.4f}, F1-score: {val_f1:.4f}"
        progressBar.set_description( progressText )
        
        printTable({"Accuracy" : val_acc,
         "Precision" : val_precision,
         "Recall" : val_recall,
         "F1 score" : val_f1
        })
        
        torch.save( model.state_dict(), os.path.join(SAVEPATH, f"model_{PurePath(directories[test[0]]).parts[-1]}_{eid}.pt") )
        
    y_pred, y_true = [], []
    with torch.no_grad():
        for val_input, val_label in val_dataloader:
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask).logits

            batch_loss = criterion(output, val_label)
            total_loss_val += batch_loss.item()

            y_pred.extend( output.argmax(dim=1).cpu().tolist() )
            y_true.extend( val_label.cpu().tolist() )

    tst_accuracy = accuracy_score( y_true, y_pred, normalize=True )
    tst_precision, tst_recall, tst_f1, _ = precision_recall_fscore_support( y_true, y_pred, average='macro') # micro

    acc_set.append( tst_accuracy )
    prec_set.append( tst_precision )
    recall_set.append( tst_recall )
    f1_set.append( tst_f1 )


    printTable({"Accuracy" : tst_accuracy,
     "Precision" : tst_precision,
     "Recall" : tst_recall,
     "F1 score" : tst_f1
    })
# -


printTable({"Accuracy" : sum(acc_set)/len(acc_set),
 "Precision" : sum(prec_set)/len(prec_set),
 "Recall" : sum(recall_set)/len(recall_set),
 "F1 score" : sum(f1_set)/len(f1_set)
})
