import time
import copy
import os
from pathlib import PurePath
from tqdm import tqdm
import json
import pickle
import random
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tabulate

import torch
from torch_geometric import utils
from torch_geometric.data import Data, DataLoader

from early_data_3label import *
from feature_extractor import *

from gcn import *
from gat import *
import sys

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(12345)
random.seed(12345)
np.random.seed(12345)

if len(sys.argv) >= 2:
    TRAIN_DATA_PATH = sys.argv[1]
else:
    TRAIN_DATA_PATH = "./data/pheme9/3label"

if len(sys.argv) >= 3:
    TEST_DATA_PATH = sys.argv[2]
else:
    TEST_DATA_PATH = "./data/pheme9/3label"

if len(sys.argv) >= 4:
    SAVE_DIR = sys.argv[3]
else:
    SAVE_DIR = "./models/3label"

if len(sys.argv) >= 5:
    MODEL = sys.argv[4]
else:
    MODEL = "GCN"

if len(sys.argv) >= 6:
    EPOCH = int(sys.argv[5])
else:
    EPOCH = 100

if len(sys.argv) >= 7:
    PHEME_DATASET = sys.argv[6]
else:
    PHEME_DATASET = "./rumoureval2019"

if len(sys.argv) >= 8:
    TWEET_FEAT_DIR = sys.argv[7]
else:
    TWEET_FEAT_DIR = "./"

with open( os.path.join( PHEME_DATASET, "final-eval-key.json" ), "r" ) as handle:
    TWEET_ID_TO_LABEL = json.load( handle )["subtaskbenglish"]  

args = {
#     "path": DATASET_PATH,
    "num_of_node_features": 772,
    "batch": 128,
    "save_dir": SAVE_DIR
}

global_feature_extractor = FEATUREEXTRACTOR("vinai/bertweet-base", "cuda")

BATCH_SIZE = int(args["batch"])
PATH = args["save_dir"]
LR = 5e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = os.path.join(PATH, MODEL)
os.makedirs(PATH, exist_ok=True)

# +
# # Get the name of all the folders in the parent folder.
# directories = [os.path.join(args["path"], o) for o in os.listdir(
#     args["path"]) if os.path.isdir(os.path.join(args["path"], o))]


# +
# kf = KFold(n_splits=len(directories))
# -

def printTable(metric_score):
    l = [[i, j] for i, j in metric_score.items()]

    table = tabulate.tabulate(
        l, headers=['Metric', 'Value'], tablefmt='orgtbl')
    print(table)


with open(os.path.join(TWEET_FEAT_DIR, "tweet_features.pickle"), "rb") as handle:
    TWEET_FEAT = pickle.load(handle)


def early_RD(model, save_dir, criterion):

    total = {}

    print("Early RD Time")
    for hour in tqdm([0.00001, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36]):
        
        pheme_data_path = os.path.join( PHEME_DATASET, "rumoureval-2019-test-data/twitter-en-test-data" )
        
        test_data = getEarlyRDTime3Label(
            hour, pheme_data_path, TWEET_FEAT, global_feature_extractor, TWEET_ID_TO_LABEL)

        for graph in test_data:
            x = torch.tensor(graph['x'], dtype=torch.float)
            e = torch.tensor(graph['edge_list'], dtype=torch.long)
            e = utils.add_self_loops(e, num_nodes=len(graph['x']))[0]
            if graph['y'] == "unverified":
                category = torch.tensor([2], dtype=torch.long)
            elif graph['y'] == "true":
                category = torch.tensor([1], dtype=torch.long)
            else:
                category = torch.tensor([0], dtype=torch.long)

            test_list.append(Data(x=x, edge_index=e, y=category))

        test_loader = DataLoader(test_list, batch_size=512)

        model.eval()
        test_loss, test_acc, test_prec, test_recall, test_f1 = model._testEarly(
            test_loader, criterion, DEVICE)

        total[hour] = {
            "loss": test_loss,
            "acc": test_acc,
            "precision": test_prec,
            "recall": test_recall,
            "f1 score": test_f1
        }

    with open(os.path.join(save_dir, f"time_result.pickle"), "wb") as handle:
        pickle.dump(total, handle, protocol=pickle.HIGHEST_PROTOCOL)


def early_RD_comment(model, save_dir, criterion):

    total = {}
    print("Early RD Comment")
    for commentLimit in tqdm([1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50]):
        
        pheme_data_path = os.path.join( PHEME_DATASET, "rumoureval-2019-test-data/twitter-en-test-data" )
        
        test_data = getEarlyRDComment3Label(
            commentLimit, pheme_data_path, TWEET_FEAT, global_feature_extractor, TWEET_ID_TO_LABEL)

        for graph in test_data:
            x = torch.tensor(graph['x'], dtype=torch.float)
            e = torch.tensor(graph['edge_list'], dtype=torch.long)
            e = utils.add_self_loops(e, num_nodes=len(graph['x']))[0]
            if graph['y'] == "unverified":
                category = torch.tensor([2], dtype=torch.long)
            elif graph['y'] == "true":
                category = torch.tensor([1], dtype=torch.long)
            else:
                category = torch.tensor([0], dtype=torch.long)

            test_list.append(Data(x=x, edge_index=e, y=category))

        test_loader = DataLoader(test_list, batch_size=512)

        model.eval()
        test_loss, test_acc, test_prec, test_recall, test_f1 = model._testEarly(
            test_loader, criterion, DEVICE)

        total[commentLimit] = {
            "loss": test_loss,
            "acc": test_acc,
            "precision": test_prec,
            "recall": test_recall,
            "f1 score": test_f1
        }

    with open(os.path.join(save_dir, f"comment_result.pickle"), "wb") as handle:
        pickle.dump(total, handle, protocol=pickle.HIGHEST_PROTOCOL)


# +
START_TIME = time.time()
AVG_RESULTS = {}
PER_EPOCH_RESULTS = {}


CURR_PATH = os.path.join(PATH, PurePath(TEST_DATA_PATH).parts[-1])
train_list, test_list = [], []

# Create folder with test event name
os.makedirs(CURR_PATH, exist_ok=True)

print("*" * 60)
print("Testing: ", PurePath(TEST_DATA_PATH).parts[-1])

avg_key = PurePath(TEST_DATA_PATH).parts[-1]

AVG_RESULTS[avg_key] = {}
AVG_RESULTS[avg_key]["loss"] = 10000000
AVG_RESULTS[avg_key]["acc"] = 0
AVG_RESULTS[avg_key]["precision"] = 0
AVG_RESULTS[avg_key]["recall"] = 0
AVG_RESULTS[avg_key]["f1score"] = 0

PER_EPOCH_RESULTS[avg_key] = []

unverified, tru, fal = 0, 0, 0

# graph_path = os.path.join( directories[ i ],  f"graph_{EXPERIMENT}.pickle")
graph_path = os.path.join(TRAIN_DATA_PATH,  f"graph.pickle")
# print(directories[i])
with open(graph_path, "rb") as input_file:
    local_g = pickle.load(input_file)

for gidx, g in enumerate(local_g):
    if g['y'] == "unverified":
        l = torch.tensor([2], dtype=torch.long)
        unverified = unverified + 1
    elif g['y'] == "true":
        l = torch.tensor([1], dtype=torch.long)
        tru = tru + 1
    else:
        l = torch.tensor([0], dtype=torch.long)
        fal = fal + 1

    x = torch.tensor(g['x'], dtype=torch.float)
    e = torch.tensor(g['edge_list'], dtype=torch.long)
    e = utils.add_self_loops(e, num_nodes=len(g['x']))[0]

    num_node_features = len(g['x'][0])

    train_list.append(Data(x=x, edge_index=e, y=l))

print(
    f"Training set contains:\nNumber of unverified tweets {unverified},  true tweets {tru}, false tweet {fal}.")

unverified, tru, fal = 0, 0, 0
# graph_path = os.path.join( directories[ i ],  f"graph_{EXPERIMENT}.pickle")
graph_path = os.path.join(TEST_DATA_PATH,  f"graph_test.pickle")
with open(graph_path, "rb") as input_file:
    local_g = pickle.load(input_file)

for gidx, g in enumerate(local_g):
    if g['y'] == "unverified":
        l = torch.tensor([2], dtype=torch.long)
        unverified = unverified + 1
    elif g['y'] == "true":
        l = torch.tensor([1], dtype=torch.long)
        tru = tru + 1
    else:
        l = torch.tensor([0], dtype=torch.long)
        fal = fal + 1

    x = torch.tensor(g['x'], dtype=torch.float)
    e = torch.tensor(g['edge_list'], dtype=torch.long)
    e = utils.add_self_loops(e, num_nodes=len(g['x']))[0]

    test_list.append(Data(x=x, edge_index=e, y=l))

print(
    f"Testing set contains:\nNumber of unverified tweets {unverified},  true tweets {tru}, false tweet {fal}.")

print("Device: ", DEVICE)

train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_list, batch_size=1)

print("\n")
print("*" * 60)
print("\nHyperparameters for the network:\n")
print("1. Number of node features: ", num_node_features)
print("2. Hidden channels: ", 1024)
print("3. Number of output classes: ", 2)
print("*" * 60)
print("\n")

if MODEL == "GCN":
    model = GCNNet(num_node_features=num_node_features,
                   hidden_channels=1024,  num_classes=3)
else:
    model = GATNet3(num_node_features=num_node_features, hidden_channels=1024,
                    heads=8, dropout=0.5, num_classes=3, training=True)

model = model.to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

BEST_LOSS = 100000
GLOBAL_LOSS, GLOBAL_VAL_LOSS, GLOBAL_TRAIN_ACC, GLOBAL_TEST_ACC = [], [], [], []

progressBar = tqdm(range(1, int(EPOCH) + 1))

BEST_MODEL = copy.deepcopy(model)

for epoch in progressBar:
    model.train()
    loss, train_acc = model._train(
        optimizer, criterion, train_loader, DEVICE)
    model.eval()
    test_loss, test_acc, test_prec, test_recall, test_f1 = model._test(
        test_loader, criterion, DEVICE)
    progressBar.set_description(
        f'Epoch: {epoch:03d}, Test Acc: {test_acc*100:.2f}, Test f1 score: {test_f1}')

    res = {
        "loss": test_loss,
        "acc": test_acc,
        "precision": test_prec,
        "recall": test_recall,
        "f1score": test_f1
    }
    if AVG_RESULTS[avg_key]["f1score"] <= test_f1:
        AVG_RESULTS[avg_key]["loss"] = test_loss
        AVG_RESULTS[avg_key]["acc"] = test_acc
        AVG_RESULTS[avg_key]["precision"] = test_prec
        AVG_RESULTS[avg_key]["recall"] = test_recall
        AVG_RESULTS[avg_key]["f1score"] = test_f1
        BEST_MODEL = copy.deepcopy(model)
        # torch.save(model, os.path.join(CURR_PATH, "model.pt"))

    PER_EPOCH_RESULTS[avg_key].append(res)
    GLOBAL_LOSS.append(loss)
    GLOBAL_VAL_LOSS.append(test_loss)
    GLOBAL_TRAIN_ACC.append(train_acc * 100)
    GLOBAL_TEST_ACC.append(test_acc * 100)

printTable(AVG_RESULTS[avg_key])
# Store the results
with open(os.path.join(CURR_PATH, "average_results.json"), "w") as jf:
    json.dump(AVG_RESULTS[avg_key], jf)

plt.clf()
# Create the loss and accuracy graph
plt.plot(GLOBAL_LOSS, label="Training loss", color='green')
plt.plot(GLOBAL_VAL_LOSS, label="Validation loss", color="red")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.savefig(os.path.join(CURR_PATH, "loss.png"))
plt.savefig(os.path.join(CURR_PATH, "loss.svg"))

plt.clf()

plt.plot(GLOBAL_TRAIN_ACC, label="Training accuracy", color='green')
plt.plot(GLOBAL_TEST_ACC, label="Validation accuracy", color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="upper right")
plt.savefig(os.path.join(CURR_PATH, "acc.png"))
plt.savefig(os.path.join(CURR_PATH, "acc.svg"))

plt.clf()

# Early RD
early_RD(BEST_MODEL, CURR_PATH, criterion)
early_RD_comment(BEST_MODEL, CURR_PATH, criterion)
# -

END_TIME = time.time()

with open(os.path.join(PATH, "per_epoch_results.json"), "w") as jf:
    json.dump(PER_EPOCH_RESULTS, jf)

acc, prec, recall, f1 = 0, 0, 0, 0
for key, val in AVG_RESULTS.items():
    acc = acc + val["acc"]
    prec = prec + val["precision"]
    recall = recall + val["recall"]
    f1 = f1 + val["f1score"]

printTable({"Accuracy": acc / 9, "Precision": prec / 9,
           "Recall": recall / 9, "F1 score": f1 / 9})

with open(os.path.join(PATH, "average_results.json"), "w") as jf:
    json.dump({"Accuracy": acc / 9, "Precision": prec / 9,
              "Recall": recall / 9, "F1 score": f1 / 9}, jf)

print("TOTAL RUN TIME: ", (END_TIME - START_TIME)/60)

# +
# total_early_time, total_early_comment = {}, {}
# for d in directories:
#     event = PurePath(d).parts[-1]

#     CURR_PATH = os.path.join(PATH, event)

#     with open(os.path.join(CURR_PATH, "time_result.pickle"), "rb") as handle:
#         time_data = pickle.load(handle)

#     with open(os.path.join(CURR_PATH, "comment_result.pickle"), "rb") as handle:
#         comment_data = pickle.load(handle)

#     for hour in tqdm([0.00001, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 36]):
#         total_early_time[hour] = {
#             "loss": 0,
#             "acc": 0,
#             "precision": 0,
#             "recall": 0,
#             "f1 score": 0
#         }

#         metrics = time_data[hour]

#         for k, v in metrics.items():
#             total_early_time[hour][k] = total_early_time[hour][k] + metrics[k]

#     for commentLimit in tqdm([1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50]):
#         total_early_comment[commentLimit] = {
#             "loss": 0,
#             "acc": 0,
#             "precision": 0,
#             "recall": 0,
#             "f1 score": 0
#         }
#         metrics = comment_data[commentLimit]
#         for k, v in metrics.items():
#             total_early_comment[commentLimit][k] = total_early_comment[commentLimit][k] + metrics[k]

# +
# for k, v in total_early_time.items():
#     for m, vv in v.items():
#         total_early_time[k][m] = total_early_time[k][m] / 9

# +
# for k, v in total_early_comment.items():
#     for m, vv in v.items():
#         total_early_comment[k][m] = total_early_comment[k][m] / 9

# +
# with open(os.path.join(PATH, f"average_time_result.pickle"), "wb") as handle:
#     pickle.dump(total_early_time, handle, protocol=pickle.HIGHEST_PROTOCOL)

# +
# with open(os.path.join(PATH, f"average_comment_result.pickle"), "wb") as handle:
#     pickle.dump(total_early_comment, handle, protocol=pickle.HIGHEST_PROTOCOL)
