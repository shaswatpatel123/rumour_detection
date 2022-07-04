import torch
import transformers
from data_augmentation import *
from util_graph import *
from util import *
import sys
import numpy as np
import random
import pickle
from nltk.tokenize import TweetTokenizer
from pathlib import PurePath
import os
from tqdm import tqdm
import json
random.seed(12345)
np.random.seed(12345)
args = {
    "path": "./all-rnr-annotated-threads",
    "save_path": os.path.join("./", "pheme")
}

# Get the name of all the folders in the parent folder.
directories = [os.path.join(args["path"], o) for o in os.listdir(args["path"]) if os.path.isdir(os.path.join(args["path"], o))]

# Traverse through it to get Source Tweet and Reaction Tweet
for dir in directories:

    topic = PurePath(dir).parts[-1].split('-')[0]
    # Traverse through rumour and non-rumour directories
    sub_dir = [os.path.join(dir, i) for i in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, i))]

    event = PurePath(dir).parts[-1].split('-')[0]
    print(f"Currently processing {event} event.")
    # Make directory for this event at the saving path
    save_path = os.path.join(args["save_path"])
    os.makedirs( os.path.join(args["save_path"], event), exist_ok=True )

    label_json = {}
    for sdir in sub_dir:
        tweets = [os.path.join(sdir, i) for i in os.listdir(
            sdir) if os.path.isdir(os.path.join(sdir, i))]
        label = PurePath(sdir).parts[-1]

        
        os.makedirs(os.path.join(save_path, event, label), exist_ok=True)

        print(f"{event} event contains {len(tweets)} tweets for {label} label")

        os.makedirs(os.path.join(save_path, event, label), exist_ok=True)

        for t in tqdm(tweets):
            id = PurePath(t).parts[-1]

            # Source folder
            source_path = os.path.join(t, "source-tweets", f"{id}.json")
            featureMatrix = []

            with open(source_path, encoding="utf8") as ofile:
                source_json = json.load(ofile)
                featureMatrix.append(source_json["text"])

            # Reaction Folder
            reaction_path = os.path.join(t, "reactions")
            reactions = [os.path.join(reaction_path, i) for i in os.listdir(
                reaction_path) if i.split('\\')[-1][0] != '.']

            for reaction in reactions:
                with open(reaction, encoding="utf8") as ofile:
                    obj = json.load(ofile)
                    featureMatrix.append(obj["text"])

            tweetData = {
                "featureMatrix": featureMatrix,
                "label": label
            }

            # Write in the saving path as id.json
            with open(os.path.join(save_path, event, label, f"{id}.json"), "w", encoding="utf8") as ofile:
                json.dump(tweetData, ofile, indent=4)

    with open(os.path.join(save_path, event, "labels.json"), "w", encoding="utf8") as ofile:
        json.dump(label_json, ofile, indent=4)

    print("\n")
    print("*" * 60)
    
SAVE_DIR = "./"
unaugmented_save_path = os.path.join(SAVE_DIR, "unaugmented")
os.makedirs(unaugmented_save_path, exist_ok=True)
for event in os.listdir(os.path.join(SAVE_DIR, "pheme")):
    print("\n")
    print("*" * 60)
    print(event)

    graphList = []

    rumourLabel = [
        os.path.join(SAVE_DIR, "pheme", event, "rumours", i) for i in os.listdir(os.path.join(SAVE_DIR, "pheme", event, "rumours"))]
    nonrumourLabel = [
        os.path.join(SAVE_DIR, "pheme", event, "non-rumours", i) for i in os.listdir(os.path.join(SAVE_DIR, "pheme", event, "non-rumours"))]

    # print(f"\nTrue : {len(trueLabel)} | False : {len(falseLabel)} | Unverified : {len(unverifiedLabel)} tweets\n")
    print(f"Rumour : {len(rumourLabel)} | Non-rumour : {len(nonrumourLabel)}")

    for x in [rumourLabel, nonrumourLabel]:
        for f in tqdm(x):
            with open(f, encoding="utf8") as ofile:
                data = json.load(ofile)
            data['x'] = '. '.join( data['featureMatrix'] )
            graphList.append( data )

    # Save the un-augmented graphList to be used as test dataset
    os.makedirs(os.path.join(unaugmented_save_path, event), exist_ok=True)
    pickle_path = os.path.join(
        unaugmented_save_path, event, 'graph_test.pickle')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_path = os.path.join(
        unaugmented_save_path, event, 'graph.pickle')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n")
    print("*" * 60)
    
print()
print("Augmented data")
print()
SAVE_DIR="./"
getAugmentedData2Label(SAVE_DIR, False)

print()
print("Improved Augmented data")
print()
SAVE_DIR="./"
getAugmentedData2Label(SAVE_DIR, True)