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

if len(sys.argv) >= 2:
    DATA_PATH = sys.argv[1]
else:
    DATA_PATH = "./all-rnr-annotated-threads"

if len(sys.argv) >= 3:
    SAVE_DIR = sys.argv[2]
else:
    SAVE_DIR = "./data/pheme9/3label"

if len(sys.argv) >= 4:
    TWEET_FEAT_DIR = sys.argv[3]
else:
    TWEET_FEAT_DIR = "./"


global_feature_extractor = FEATUREEXTRACTOR("vinai/bertweet-base", "cuda")

with open(os.path.join(TWEET_FEAT_DIR, "tweet_features.pickle"), "rb") as handle:
    TWEET_FEAT = pickle.load(handle)

args = {
    "path": DATA_PATH,
    "save_path": os.path.join(SAVE_DIR, "pheme")
}

# Get the name of all the folders in the parent folder.
directories = [os.path.join(args["path"], o) for o in os.listdir(
    args["path"]) if os.path.isdir(os.path.join(args["path"], o))]

print("*" * 60)
print("Number of Events: ", len(directories))
for idx, i in enumerate(directories):
    print(f"{(idx + 1)}. {PurePath(i).parts[-1].split('-')[0]}")
print("*" * 60)

feature_extractor = FEATUREEXTRACTOR("vinai/bertweet-base", "cuda")

# Traverse through it to get Source Tweet and Reaction Tweet
data = {}
for dir in directories:

    topic = PurePath(dir).parts[-1].split('-')[0]

    data[topic] = {}
    # Traverse through rumour and non-rumour directories
    sub_dir = [os.path.join(dir, i) for i in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, i))]

    event = PurePath(dir).parts[-1].split('-')[0]
    print(f"Currently processing {event} event.")
    # Make directory for this event at the saving path
    save_path = os.path.join(args["save_path"])

    os.makedirs(os.path.join(save_path, event), exist_ok=True)
    label_json = {}
    for sdir in sub_dir:
        tweets = [os.path.join(sdir, i) for i in os.listdir(
            sdir) if os.path.isdir(os.path.join(sdir, i))]
        label = PurePath(sdir).parts[-1]

        if label == "non-rumours":
            continue

        # Make directory for this label inside the event
        os.makedirs(os.path.join(save_path, event, label), exist_ok=True)
        print(f"{event} event contains {len(tweets)} tweets for {label} label")

        os.makedirs(os.path.join(save_path, event,
                    label, "true"), exist_ok=True)
        os.makedirs(os.path.join(save_path, event,
                    label, "false"), exist_ok=True)
        os.makedirs(os.path.join(save_path, event,
                    label, "unverified"), exist_ok=True)
        for t in tqdm(tweets):
            id = PurePath(t).parts[-1]

            # Structure
            structure_json = getStructure(t)

            edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map(
                structure_json, id)

            # Annotations
            real_label = getAnnotation(t)
            label_json[id] = real_label

            tweetData = {
                "edgeList": edgeList,
                "nodeToIndexMap": nodeToIndexMap,
                "nodeToIndexMapArray": nodeToIndexMapArray,
                "label": real_label
            }

            # Feature Matrix
            # Source folder
            source_json = getSource(t, id)
            # Reaction Folder
            reaction_json = getReactions(t)

            tweetData["featureMatrix"], tweetData["tweetIDList"] = create_feature_matrix(
                source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray, global_feature_extractor)

            # Write in the saving path as id.json
            with open(os.path.join(save_path, event, label, real_label, f"{id}.json"), "w", encoding="utf8") as ofile:
                json.dump(tweetData, ofile, indent=4)

    with open(os.path.join(save_path, event, "labels.json"), "w", encoding="utf8") as ofile:
        json.dump(label_json, ofile, indent=4)

    print("\n")
    print("*" * 60)


print()
print("Unaugmented data")
print()
data = {}
unaugmented_save_path = os.path.join(SAVE_DIR, "unaugmented")
os.makedirs(unaugmented_save_path, exist_ok=True)
for event in os.listdir(os.path.join(SAVE_DIR, "pheme")):
    print("\n")
    print("*" * 60)
    print(event)

    graphList = []

    trueLabel = [
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "true", i) for i in os.listdir(
            os.path.join(SAVE_DIR, "pheme", event, "rumours", "true"))]
    falseLabel = [os.path.join(SAVE_DIR, "pheme", event, "rumours", "false", i) for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "false"))]

    unverifiedLabel = [os.path.join(SAVE_DIR, "pheme", event, "rumours", "unverified", i) for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "unverified"))]

    print(
        f"\nTrue : {len(trueLabel)} | False : {len(falseLabel)} | Unverified : {len(unverifiedLabel)} tweets\n")

    for x in [trueLabel, falseLabel, unverifiedLabel]:
        for f in x:
            with open(f, encoding="utf8") as ofile:
                data = json.load(ofile)

            graphList.append(getGData(data, TWEET_FEAT))

    os.makedirs(os.path.join(unaugmented_save_path, event), exist_ok=True)
    # Save the un-augmented graphList to be used as test dataset

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
getAugmentedData3Label(SAVE_DIR, TWEET_FEAT, False, global_feature_extractor)


print()
print("Improved Augmented data")
print()
getAugmentedData3Label(SAVE_DIR, TWEET_FEAT, True, global_feature_extractor)
