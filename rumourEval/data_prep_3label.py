import os
from pathlib import PurePath
from tqdm import tqdm
import json
import pickle
import random
import numpy as np

import nlpaug.augmenter.word as naw
import sys

from util import *
from util_graph import *
from feature_extractor import *
from data_augmentation import *

random.seed(12345)
np.random.seed(12345)

# if len(sys.argv) >= 2:
#     DATA_PATH = sys.argv[1]
# else:
#     DATA_PATH = "./rumoureval2019"
DATA_PATH = "./rumoureval2019"

# if len(sys.argv) >= 3:
#     SAVE_DIR = sys.argv[2]
# else:
#     SAVE_DIR = "./data/"
SAVE_DIR = "./data/"

# if len(sys.argv) >= 4:
#     TWEET_FEAT_DIR = sys.argv[3]
# else:
#     TWEET_FEAT_DIR = "./"
TWEET_FEAT_DIR = "./"


# +
with open( os.path.join( DATA_PATH, "rumoureval-2019-training-data", "train-key.json" ), "r" ) as handle:
    TWEET_ID_TO_LABEL = json.load( handle )["subtaskbenglish"]

with open( os.path.join( DATA_PATH, "rumoureval-2019-training-data", "dev-key.json" ), "r" ) as handle:
    TWEET_ID_TO_LABEL = {**json.load( handle )["subtaskbenglish"], **TWEET_ID_TO_LABEL}
    
with open( os.path.join(DATA_PATH, "final-eval-key.json" ), "r" ) as handle:
    TWEET_ID_TO_LABEL = {**json.load( handle )["subtaskbenglish"], **TWEET_ID_TO_LABEL}    
# -

global_feature_extractor = FEATUREEXTRACTOR("vinai/bertweet-base", "cuda")

with open(os.path.join(TWEET_FEAT_DIR, "tweet_features.pickle"), "rb") as handle:
    TWEET_FEAT = pickle.load(handle)

args = {
    "path": DATA_PATH,
    "save_path": os.path.join(SAVE_DIR, "pheme")
}

# +
# # Get the name of all the folders in the parent folder.
# directories = [os.path.join(args["path"], o) for o in os.listdir(
#     args["path"]) if os.path.isdir(os.path.join(args["path"], o))]

# load all training data
train_data_path = os.path.join( args["path"], "rumoureval-2019-training-data", "twitter-english")
directories = [os.path.join(train_data_path, o) for o in os.listdir(
    train_data_path) if os.path.isdir(os.path.join(train_data_path, o))]
# -

print("*" * 60)
print("Number of Events: ", len(directories))
for idx, i in enumerate(directories):
    print(f"{(idx + 1)}. {PurePath(i).parts[-1].split('-')[0]}")
print("*" * 60)


def getLabel( id ):
    return TWEET_ID_TO_LABEL[ id ]


# +
# Traverse through it to get Source Tweet and Reaction Tweet
data = {}
for dir in directories[2:]:
    event = PurePath(dir).parts[-1].split('-')[0]
    print(f"Currently processing {event} event.")
    
    data[event] = {}
    
    threads = [os.path.join(dir, i) for i in os.listdir(dir) if os.path.isdir(os.path.join(dir, i))]
    
    # Make directory for this event at the saving path
    save_path = os.path.join(args["save_path"])
    
    os.makedirs(os.path.join(save_path, event, "true"), exist_ok=True)
    os.makedirs(os.path.join(save_path, event, "false"), exist_ok=True)
    os.makedirs(os.path.join(save_path, event, "unverified"), exist_ok=True)
    
    counter = 0
    
    for t in tqdm(threads):
        id = PurePath(t).parts[-1]
        
        label = getLabel( id )
        
        # Structure
        structure_json = getStructure(t)


        
        # Source folder
        source_json = getSource(t, id)
        
#         print("Source: ", id)
        
        # Reaction Folder
        reaction_json = getReactions(t)
        
        edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map(structure_json, id, source_json, reaction_json)

        tweetData = {
            "edgeList": edgeList,
            "nodeToIndexMap": nodeToIndexMap,
            "nodeToIndexMapArray": nodeToIndexMapArray,
            "label": label
        }
        
        tweetData["featureMatrix"], tweetData["tweetIDList"] = create_feature_matrix(
        source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray, global_feature_extractor)
        
        # Write in the saving path as id.json
        with open(os.path.join(save_path, event, label, f"{id}.json"), "w", encoding="utf8") as ofile:
            json.dump(tweetData, ofile, indent=4)
                
    print("\n")
    print("*" * 60)

# +
print()
print("Unaugmented data")
print()
# data = {}
graphList = []
unaugmented_save_path = os.path.join(SAVE_DIR, "unaugmented")
os.makedirs(unaugmented_save_path, exist_ok=True)
for event in os.listdir(os.path.join(SAVE_DIR, "pheme")):
    print("\n")
    print("*" * 60)
    print(event)

    trueLabel = [
        os.path.join(SAVE_DIR, "pheme", event, "true", i) for i in os.listdir(
            os.path.join(SAVE_DIR, "pheme", event, "true"))]
    falseLabel = [os.path.join(SAVE_DIR, "pheme", event, "false", i) for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "false"))]

    unverifiedLabel = [os.path.join(SAVE_DIR, "pheme", event, "unverified", i) for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "unverified"))]

    print(
        f"\nTrue : {len(trueLabel)} | False : {len(falseLabel)} | Unverified : {len(unverifiedLabel)} tweets\n")

    for x in [trueLabel, falseLabel, unverifiedLabel]:
        for f in x:
            with open(f, encoding="utf8") as ofile:
                data = json.load(ofile)

            graphList.append(getGData(data, TWEET_FEAT))

    # os.makedirs(os.path.join(unaugmented_save_path, event), exist_ok=True)
    print("\n")
    print("*" * 60)
    
pickle_path = os.path.join(unaugmented_save_path, 'graph.pickle')
with open(pickle_path, 'wb') as handle:
    pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -


print()
print("Augmented data")
print()
getAugmentedData3Label(SAVE_DIR, TWEET_FEAT, False, global_feature_extractor)


print()
print("Improved Augmented data")
print()
getAugmentedData3Label(SAVE_DIR, TWEET_FEAT, True, global_feature_extractor)



# +
# Testing data generation code
# -

args = {
    "path": DATA_PATH,
    "save_path": os.path.join(SAVE_DIR, "pheme_test")
}

# +
# # Get the name of all the folders in the parent folder.
# directories = [os.path.join(args["path"], o) for o in os.listdir(
#     args["path"]) if os.path.isdir(os.path.join(args["path"], o))]

# load all training data
test_data_path = os.path.join( args["path"], "rumoureval-2019-test-data", "twitter-en-test-data")
directories = [os.path.join(test_data_path, o) for o in os.listdir(test_data_path) if os.path.isdir(os.path.join(test_data_path, o))]
# -

print("*" * 60)
print("Number of Events: ", len(directories))
for idx, i in enumerate(directories):
    print(f"{(idx + 1)}. {PurePath(i).parts[-1].split('-')[0]}")
print("*" * 60)

# Traverse through it to get Source Tweet and Reaction Tweet
data = {}
for dir in directories[2:]:
    event = PurePath(dir).parts[-1].split('-')[0]
    print(f"Currently processing {event} event.")
    
    data[event] = {}
    
    threads = [os.path.join(dir, i) for i in os.listdir(dir) if os.path.isdir(os.path.join(dir, i))]
    
    # Make directory for this event at the saving path
    save_path = os.path.join(args["save_path"])
    
    os.makedirs(os.path.join(save_path, event, "true"), exist_ok=True)
    os.makedirs(os.path.join(save_path, event, "false"), exist_ok=True)
    os.makedirs(os.path.join(save_path, event, "unverified"), exist_ok=True)
    
    counter = 0
    
    for t in tqdm(threads):
        id = PurePath(t).parts[-1]
        
        label = getLabel( id )
        
        # Structure
        structure_json = getStructure(t)


        
        # Source folder
        source_json = getSource(t, id)
        
        # Reaction Folder
        reaction_json = getReactions(t)
        
        edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map(structure_json, id, source_json, reaction_json)

        tweetData = {
            "edgeList": edgeList,
            "nodeToIndexMap": nodeToIndexMap,
            "nodeToIndexMapArray": nodeToIndexMapArray,
            "label": label
        }
        
        tweetData["featureMatrix"], tweetData["tweetIDList"] = create_feature_matrix(
        source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray, global_feature_extractor)
        
        # Write in the saving path as id.json
        with open(os.path.join(save_path, event, label, f"{id}.json"), "w", encoding="utf8") as ofile:
            json.dump(tweetData, ofile, indent=4)
                
    print("\n")
    print("*" * 60)

# +
print()
print("Unaugmented data")
print()
data = {}
graphList = []
unaugmented_save_path = os.path.join(SAVE_DIR, "unaugmented")
os.makedirs(unaugmented_save_path, exist_ok=True)
for event in os.listdir(os.path.join(SAVE_DIR, "pheme_test")):
    print("\n")
    print("*" * 60)
    print(event)

    trueLabel = [
        os.path.join(SAVE_DIR, "pheme_test", event, "true", i) for i in os.listdir(
            os.path.join(SAVE_DIR, "pheme_test", event, "true"))]
    falseLabel = [os.path.join(SAVE_DIR, "pheme_test", event, "false", i) for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme_test", event, "false"))]

    unverifiedLabel = [os.path.join(SAVE_DIR, "pheme_test", event, "unverified", i) for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme_test", event, "unverified"))]

    print(
        f"\nTrue : {len(trueLabel)} | False : {len(falseLabel)} | Unverified : {len(unverifiedLabel)} tweets\n")

    for x in [trueLabel, falseLabel, unverifiedLabel]:
        for f in x:
            with open(f, encoding="utf8") as ofile:
                data = json.load(ofile)

            graphList.append(getGData(data, TWEET_FEAT))

    # os.makedirs(os.path.join(unaugmented_save_path, event), exist_ok=True)
    
    print("\n")
    print("*" * 60)
    
pickle_path = os.path.join(unaugmented_save_path, 'graph_test.pickle')
with open(pickle_path, 'wb') as handle:
    pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -



