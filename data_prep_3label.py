from datetime import datetime
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re
from math import ceil, floor, log10
import os
from pathlib import PurePath
from tqdm import tqdm
import json
import pickle
import random
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer

import nlpaug.augmenter.word as naw
import copy
import sys

random.seed(12345)
np.random.seed(12345)

tokenizer = TweetTokenizer()

if len(sys.argv) >= 2:
    DATA_PATH = sys.argv[1]
else:
    DATA_PATH = "./all-rnr-annotated-threads"

if len(sys.argv) >= 3:
    SAVE_DIR = sys.argv[2]
else:
    SAVE_DIR = "./data/pheme9"

if len(sys.argv) >= 4:
    TWEET_FEAT_DIR = sys.argv[3]
else:
    TWEET_FEAT_DIR = "./data/pheme9"


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace(
        "n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace(
        "'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(
        " p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    return " ".join(normTweet.split())


class FEATUREEXTRACTOR:
    def __init__(self, model_path, device):
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False)

    def get_bert_features(self, tweets):
        tweets = normalizeTweet(tweets)
        input_ids = torch.tensor([self.tokenizer.encode(
            tweets, padding=True, truncation=True)]).to(self.device)
        with torch.no_grad():
            # Return [ 1 x len(input_ids) x 768 ] tensor
            features = self.model(input_ids).last_hidden_state.cpu()

        # features = torch.mean(features, dim = 2) # Converts the features into [1 x len(input_ids)] tensor
        features = features.squeeze()[0]  # Take CLS token embedding
        features = features.tolist()  # Convert to list
        return features

    def get_social_features(self, meta_data):
        """
                Return a list containing social features, namely:
                1. Tweet count: ceil( log_10(statuses_count) )
                2. Listed Count: ceil( log_10(listed_count) )
                3. Follow Ratio: : floor( log10 (#followers/#following) )
                4. Verified: True/False (1 / 0)
        """
        if int(meta_data['statuses_count']) != 0:
            statuses_count = ceil(log10(int(meta_data['statuses_count'])))
        else:
            statuses_count = 0

        if int(meta_data['listed_count']) != 0:
            listed_count = ceil(log10(int(meta_data['listed_count'])))
        else:
            listed_count = 0

        if int(meta_data['followers_count']) != 0 and int(meta_data['friends_count']) != 0:
            follow_ratio = floor(
                log10(int(meta_data['followers_count']) / int(meta_data['friends_count'])))
        else:
            follow_ratio = 0

        verified = int(meta_data['verified'])

        features = [statuses_count, listed_count, follow_ratio, verified]
        return features

    def __extract__(self, json):

        # Text features
        text_features = self.get_bert_features(json['text'])

        # Social features
        social_features = self.get_social_features(json['user'])

        return text_features, social_features


feature_extractor = FEATUREEXTRACTOR("vinai/bertweet-base", "cuda")

with open(os.path.join(TWEET_FEAT_DIR, "tweet_features.pickle"), "rb") as handle:
    TWEET_FEAT = pickle.load(handle)


def convert_annotations(annotation, string=True):
    if 'misinformation' in annotation.keys() and 'true' in annotation.keys():
        if int(annotation['misinformation']) == 0 and int(annotation['true']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 0 and int(annotation['true']) == 1:
            if string:
                label = "true"
            else:
                label = 1
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 0:
            if string:
                label = "false"
            else:
                label = 0
        elif int(annotation['misinformation']) == 1 and int(annotation['true']) == 1:
            print("OMG! They both are 1!")
            print(annotation['misinformation'])
            print(annotation['true'])
            label = None

    elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
        # all instances have misinfo label but don't have true label
        if int(annotation['misinformation']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 1:
            if string:
                label = "false"
            else:
                label = 0

    elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
        print('Has true not misinformation')
        label = None
    else:
        label = "nonrumour"

    return label


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


def create_graph_with_map(structure_json, id):
    structure_json = {id: structure_json[id]}
    # edgeList = [[],[]]
    edgeList = []
    nodeToIndexMap = {'/': -1}
    nodeToIndexMapArray = []
    currentIndex = 0
    stack = [('/', structure_json)]
    firstFlag = True
    while(len(stack) > 0):
        currentNode, context = stack.pop()
        if not isinstance(context, list):
            for comment in context.keys():
                nodeToIndexMap[comment] = currentIndex
                nodeToIndexMapArray.append(comment)
                edgeList.append((nodeToIndexMap[currentNode], currentIndex))
                currentIndex = currentIndex + 1
                stack.append((comment, context[comment]))
    del nodeToIndexMap['/']
    edgeList.pop(0)

    return edgeList, nodeToIndexMap, nodeToIndexMapArray


def get_features(tweet_json, tweet_id):
    tweet_text = tweet_json["text"]
    user_feature = tweet_json["user"]
    social_features = feature_extractor.get_social_features(user_feature)
    return [tweet_text, *social_features]


def create_feature_matrix(source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray):
    tweets = source_json
    tweets.update(reaction_json)
    feature_matrix = []
    tweet_id_list = []
    for tweet_id in nodeToIndexMapArray:
        feature_matrix.append(get_features(tweets[tweet_id], tweet_id))
        tweet_id_list.append(tweet_id)
    return feature_matrix, tweet_id_list


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
            structure_path = os.path.join(t, 'structure.json')
            with open(structure_path, encoding="utf8") as ofile:
                content = ofile.read()
                clean = content.replace('”', '"')
                structure_json = json.loads(clean)

            edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map(
                structure_json, id)

            # Annotations
            annotation_path = os.path.join(t, 'annotation.json')
            with open(annotation_path, encoding="utf8") as ofile:
                annotation_json = json.load(ofile)
                real_label = convert_annotations(annotation_json)
            label_json[id] = real_label

            tweetData = {
                "edgeList": edgeList,
                "nodeToIndexMap": nodeToIndexMap,
                "nodeToIndexMapArray": nodeToIndexMapArray,
                "label": real_label
            }

            # Feature Matrix
            # Source folder
            source_json = {}
            source_path = os.path.join(t, "source-tweets", f"{id}.json")

            with open(source_path, encoding="utf8") as ofile:
                source_json[id] = json.load(ofile)

            # Reaction Folder
            reaction_json = {}

            reaction_path = os.path.join(t, "reactions")
            reactions = [os.path.join(reaction_path, i) for i in os.listdir(
                reaction_path) if i.split('\\')[-1][0] != '.']

            for reaction in reactions:
                with open(reaction, encoding="utf8") as ofile:
                    obj = json.load(ofile)
                reaction_id = PurePath(reaction).parts[-1].split('.')[0]

                reaction_json[reaction_id] = obj

            tweetData["featureMatrix"], tweetData["tweetIDList"] = create_feature_matrix(
                source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray)

            # Write in the saving path as id.json
            with open(os.path.join(save_path, event, label, real_label, f"{id}.json"), "w", encoding="utf8") as ofile:
                json.dump(tweetData, ofile, indent=4)

    with open(os.path.join(save_path, event, "labels.json"), "w", encoding="utf8") as ofile:
        json.dump(label_json, ofile, indent=4)

    print("\n")
    print("*" * 60)


def getTemporalFeatures(parentNode, childNode):
    """
      parentNode : tweet Obj.
      childNode : tweet Obj.
      Return
        1D list : 1st value is tweet created date diff and 2nd value is user created date diff
    """

    parentTweetTime = datetime.strptime(
        parentNode["created_at"], '%a %b %d %H:%M:%S %z %Y')
    childTweetTime = datetime.strptime(
        childNode["created_at"], '%a %b %d %H:%M:%S %z %Y')

    return (childTweetTime - parentTweetTime).total_seconds()


def create_graph_with_map_timeoffset(structure_json, id, source_json, reaction_json, time_limit):
    combined_json = {**source_json, **reaction_json}

    structure_json = {id: structure_json[id]}
    edgeList = []
    nodeToIndexMap = {'/': -1}
    nodeToIndexMapArray = []
    currentIndex = 0

    sTime = datetime.strptime(
        combined_json[id]["created_at"], '%a %b %d %H:%M:%S %z %Y').timestamp()

    stack = [('/', structure_json, 0)]
    firstFlag = True
    while(len(stack) > 0):
        currentNode, context, timeLapsed = stack.pop()
        if not isinstance(context, list):
            for comment in context.keys():
                if currentNode != "/":
                    edgeTimeDiff = getTemporalFeatures(
                        combined_json[currentNode], combined_json[comment])
                else:
                    edgeTimeDiff = 0

                if timeLapsed + edgeTimeDiff > time_limit:
                    continue

                nodeToIndexMap[comment] = currentIndex
                nodeToIndexMapArray.append(comment)
                edgeList.append((nodeToIndexMap[currentNode], currentIndex))
                currentIndex = currentIndex + 1
                stack.append(
                    (comment, context[comment], timeLapsed + edgeTimeDiff))
    del nodeToIndexMap['/']
    edgeList.pop(0)

    return edgeList, nodeToIndexMap, nodeToIndexMapArray


def create_graph_with_map_comment(structure_json, id, source_json, reaction_json, comment_limit):
    combined_json = {**source_json, **reaction_json}
    structure_json = {id: structure_json[id]}
    # edgeList = [[],[]]
    edgeList = []
    nodeToIndexMap = {'/': -1}
    nodeToIndexMapArray = []
    currentIndex = 0

    stack = [('/', structure_json)]
    firstFlag = True

    while(len(stack) > 0):
        currentNode, context = stack.pop()
        if not isinstance(context, list):
            for comment in context.keys():
                if comment_limit <= currentIndex:
                    continue
                nodeToIndexMap[comment] = currentIndex
                nodeToIndexMapArray.append(comment)
                edgeList.append((nodeToIndexMap[currentNode], currentIndex))
                currentIndex = currentIndex + 1
                stack.append((comment, context[comment]))
    del nodeToIndexMap['/']
    edgeList.pop(0)

    return edgeList, nodeToIndexMap, nodeToIndexMapArray


# Traverse through it to get Source Tweet and Reaction Tweet
master = {}
time_save_path = os.path.join(SAVE_DIR, "time.pickle")
for timiLimit in [0.00001, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 36]:
    print(timiLimit, " Hour")
    timeName = str(timiLimit)
    timiLimit = timiLimit*3600
    data = {}
    master[timiLimit] = {}

    for dir in directories:

        topic = PurePath(dir).parts[-1].split('-')[0]
        master[timiLimit][topic] = []

        data[topic] = {}
        # Traverse through rumour and non-rumour directories
        sub_dir = [os.path.join(dir, i) for i in os.listdir(
            dir) if os.path.isdir(os.path.join(dir, i))]

        event = PurePath(dir).parts[-1].split('-')[0]
        print(f"Currently processing {event} event.")
        # Make directory for this event at the saving path
        save_path = os.path.join(args["save_path"])

        for sdir in sub_dir:
            label = PurePath(sdir).parts[-1]
            if label == "non-rumours":
                continue

            tweets = [os.path.join(sdir, i) for i in os.listdir(
                sdir) if os.path.isdir(os.path.join(sdir, i))]

            print(f"{event} event contains {len(tweets)} tweets for {label} label")

            for t in tqdm(tweets):
                id = PurePath(t).parts[-1]

                # Structure
                structure_path = os.path.join(t, 'structure.json')
                with open(structure_path, encoding="utf8") as ofile:
                    content = ofile.read()
                    clean = content.replace('”', '"')
                    # json_data = json.loads(clean)
                    structure_json = json.loads(clean)

                # Annotations
                annotation_path = os.path.join(t, 'annotation.json')
                with open(annotation_path, encoding="utf8") as ofile:
                    annotation_json = json.load(ofile)
                    real_label = convert_annotations(annotation_json)

                # Source folder
                source_json = {}
                source_path = os.path.join(t, "source-tweets", f"{id}.json")
                with open(source_path, encoding="utf8") as ofile:
                    source_json[id] = json.load(ofile)

                # Reaction Folder
                reaction_json = {}
                reaction_path = os.path.join(t, "reactions")
                reactions = [os.path.join(reaction_path, i) for i in os.listdir(
                    reaction_path) if i.split('\\')[-1][0] != '.']

                for reaction in reactions:
                    with open(reaction, encoding="utf8") as ofile:
                        obj = json.load(ofile)

                    reaction_id = PurePath(reaction).parts[-1].split('.')[0]

                    reaction_json[reaction_id] = obj

                edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map_timeoffset(
                    structure_json, id, source_json, reaction_json, timiLimit)
                featureMatrix, tweetMatrix = create_feature_matrix(
                    source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray)

                features = []
                for i, j in enumerate(featureMatrix):
                    # local_features = []
                    # local_features.extend(feature_extractor.get_bert_features(j[0]))
                    local_features = TWEET_FEAT[tweetMatrix[i]]
                    local_features.extend(j[1:])  # Social features
                    features.append(local_features)

                gdata = {}
                gdata['x'] = features
                gdata['y'] = real_label
                gdata['edge_list'] = np.array(edgeList).T.tolist()

                master[timiLimit][topic].append(gdata)
        print("\n")
        print("*" * 60)

with open(os.path.join(time_save_path), "wb") as handle:
    pickle.dump(master, handle)

# Traverse through it to get Source Tweet and Reaction Tweet
comment_list = [i for i in range(2, 52, 2)]
comment_list.append(1)
master = {}
comment_save_path = os.path.join(SAVE_DIR, "comments.pickle")
for commentLimit in comment_list:
    print(commentLimit, " Comments")
    commentName = str(commentLimit)
    data = {}
    master[commentLimit] = {}

    for dir in directories:

        topic = PurePath(dir).parts[-1].split('-')[0]
        master[commentLimit][topic] = []

        data[topic] = {}
        # Traverse through rumour and non-rumour directories
        sub_dir = [os.path.join(dir, i) for i in os.listdir(
            dir) if os.path.isdir(os.path.join(dir, i))]

        event = PurePath(dir).parts[-1].split('-')[0]
        print(f"Currently processing {event} event.")
        # Make directory for this event at the saving path
        save_path = os.path.join(args["save_path"])

        label_json = {}
        for sdir in sub_dir:
            label = PurePath(sdir).parts[-1]
            if label == "non-rumours":
                continue

            tweets = [os.path.join(sdir, i) for i in os.listdir(
                sdir) if os.path.isdir(os.path.join(sdir, i))]

            print(f"{event} event contains {len(tweets)} tweets for {label} label")

            for t in tqdm(tweets):
                id = PurePath(t).parts[-1]

                # Structure
                structure_path = os.path.join(t, 'structure.json')
                with open(structure_path, encoding="utf8") as ofile:
                    content = ofile.read()
                    clean = content.replace('”', '"')
                    # json_data = json.loads(clean)
                    structure_json = json.loads(clean)

                # Annotations
                annotation_path = os.path.join(t, 'annotation.json')
                with open(annotation_path, encoding="utf8") as ofile:
                    annotation_json = json.load(ofile)
                    real_label = convert_annotations(annotation_json)
                label_json[id] = real_label

                # Source folder
                source_json = {}
                source_path = os.path.join(t, "source-tweets", f"{id}.json")

                with open(source_path, encoding="utf8") as ofile:
                    source_json[id] = json.load(ofile)

                # Reaction Folder
                reaction_json = {}

                reaction_path = os.path.join(t, "reactions")
                reactions = [os.path.join(reaction_path, i) for i in os.listdir(
                    reaction_path) if i.split('\\')[-1][0] != '.']

                for reaction in reactions:
                    with open(reaction, encoding="utf8") as ofile:
                        obj = json.load(ofile)
                    # reaction_id=reaction.split('\\')[-1].split('.')[0]
                    reaction_id = PurePath(reaction).parts[-1].split('.')[0]

                    reaction_json[reaction_id] = obj

                edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map_comment(
                    structure_json, id, source_json, reaction_json, commentLimit)
                featureMatrix, tweetMatrix = create_feature_matrix(
                    source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray)

                features = []
                for i, j in enumerate(featureMatrix):
                    # local_features = []
                    # local_features.extend(feature_extractor.get_bert_features(j[0]))
                    local_features = TWEET_FEAT[tweetMatrix[i]]
                    local_features.extend(j[1:])  # Social features
                    features.append(local_features)

                gdata = {}
                gdata['x'] = features
                gdata['y'] = real_label
                gdata['edge_list'] = np.array(edgeList).T.tolist()

                master[commentLimit][topic].append(gdata)
        print("\n")
        print("*" * 60)

with open(os.path.join(comment_save_path), "wb") as handle:
    pickle.dump(master, handle)


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
            gdata = {}
            with open(f, encoding="utf8") as ofile:
                data = json.load(ofile)

            edge_list = data["edgeList"]
            edge_list = np.array(edge_list).T.tolist()

            features = []
            tweetMatrix = data["tweetIDList"]
            for i, j in enumerate(data['featureMatrix']):
                local_features = TWEET_FEAT[tweetMatrix[i]]
                local_features.extend(j[1:])  # Social features
                features.append(local_features)

            gdata['x'] = features
            gdata['y'] = data["label"]
            gdata['edge_list'] = edge_list

            graphList.append(gdata)

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


def getGData(tweet_data, augmented_data):
    gdata = {}
    features = []
    tweetMatrix = data["tweetIDList"]
    for i, j in enumerate(data['featureMatrix']):
        local_features = TWEET_FEAT[tweetMatrix[i]]
        local_features.extend(j[1:])  # Social features
        features.append(local_features)

    gdata['x'] = features
    gdata['y'] = tweet_data['label']

    edge_list = tweet_data["edgeList"]
    edge_list = np.array(edge_list).T.tolist()
    gdata['edge_list'] = edge_list

    return gdata


aug = naw.ContextualWordEmbsAug(
    model_path='vinai/bertweet-base', aug_p=0.30, device="cuda", stopwords=["@USER", "HTTPURL"])


def nlpAugmentation(json_file, num, p_commets_aug=0.15):
    res = []

    with open(json_file, encoding="utf8") as ofile:
        data = json.load(ofile)

    # Extract 0.15% of comments to augment
    for i, j in enumerate(data["featureMatrix"]):
        # normalize the tweets before augmentation => Prevents <unk> due to hashtags, mentions and urls
        data["featureMatrix"][i][0] = normalizeTweet(j[0])

    # All index except source
    feature_index = [i for i in range(1, len(data["featureMatrix"]))]

    # How many comments to be selected for augmentation
    k = int(ceil(len(feature_index) * p_commets_aug))

    # Choose comments
    feature_index_choosen = []
    for _ in range(num):
        feature_index_choosen.extend(np.random.choice(
            feature_index, k, replace=False).tolist())

    tweets = [data["featureMatrix"][i][0] for i in feature_index_choosen]

    augmented_tweets = aug.augment(tweets)  # All the comments
    source_augmented_twetes = aug.augment(
        data["featureMatrix"][0][0], n=num)  # Source

    augmented_tweets = np.array(augmented_tweets).reshape(
        num, int(ceil(len(feature_index) * p_commets_aug)))
    feature_index_choosen = np.array(feature_index_choosen).reshape(
        num, int(ceil(len(feature_index) * p_commets_aug)))

    for f_idx, aug_source_tweet, aug_tweet in zip(feature_index_choosen, source_augmented_twetes, augmented_tweets,):
        tmp = copy.deepcopy(data)
        for i, j in zip(f_idx, aug_tweet):
            tmp["featureMatrix"][i][0] = j
        tmp["featureMatrix"][0][0] = aug_source_tweet
        res.append(getGData(data, tmp))
    return res


def dataAugmentation(data_list, num, rem, p_commets_aug=0.15):
    graphList = []
    if num >= 1:
        print("NUM")
        for tweet in tqdm(data_list):
            augmented_data = nlpAugmentation(tweet, num, p_commets_aug)
            graphList.extend(augmented_data)

    rem_data = random.sample(data_list, rem)
    print("REM")
    for tweet in tqdm(rem_data):
        augmented_data = nlpAugmentation(tweet, 1, p_commets_aug)
        graphList.extend(augmented_data)

    return graphList


AUG_PERC = 0.15
augmented_save_path = os.path.join(SAVE_DIR, "augmented")
os.makedirs(augmented_save_path, exist_ok=True)
data = {}
for event in os.listdir(os.path.join(SAVE_DIR, "pheme")):
    print("\n")
    print("*" * 60)
    print(event)

    graphList = []

    trueLabel = [
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "true", i)
        for i in os.listdir(
            os.path.join(SAVE_DIR, "pheme", event, "rumours", "true"))]
    falseLabel = [os.path.join(SAVE_DIR, "pheme", event, "rumours", "false", i)for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "false"))]

    unverifiedLabel = [os.path.join(SAVE_DIR, "pheme", event, "rumours", "unverified", i)
                       for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "unverified"))]

    print(
        f"\nTrue : {len(trueLabel)} | False : {len(falseLabel)} | Unverified : {len(unverifiedLabel)} tweets\n")

    for x in [trueLabel, falseLabel, unverifiedLabel]:
        for f in x:
            gdata = {}
            with open(f, encoding="utf8") as ofile:
                data = json.load(ofile)

            edge_list = data["edgeList"]
            edge_list = np.array(edge_list).T.tolist()

            features = []
            tweetMatrix = data["tweetIDList"]
            for i, j in enumerate(data['featureMatrix']):
                local_features = TWEET_FEAT[tweetMatrix[i]]
                local_features.extend(j[1:])  # Social features
                features.append(local_features)

            gdata['x'] = features
            gdata['y'] = data["label"]
            gdata['edge_list'] = edge_list

            graphList.append(gdata)

    # Save the un-augmented graphList to be used as test dataset
    os.makedirs(os.path.join(augmented_save_path, event), exist_ok=True)
    pickle_path = os.path.join(augmented_save_path, event, 'graph_test.pickle')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)

    maximum = max(len(trueLabel), len(falseLabel), len(unverifiedLabel))

    if len(trueLabel) < maximum and len(trueLabel) != 0:
        num = floor(maximum / len(trueLabel))
        rem = maximum - (len(trueLabel) * int(num))

        print(
            f"\nAugmenting entire True labelled tweets {num - 1} and randomly audmenting {rem} tweets")

        graphList.extend(dataAugmentation(trueLabel, num - 1, rem, AUG_PERC))

    if len(falseLabel) < maximum and len(falseLabel) != 0:
        num = floor(maximum / len(falseLabel))
        rem = maximum - (len(falseLabel) * int(num))

        print(
            f"\nAugmenting entire False labelled tweets {num - 1} and randomly audmenting {rem} tweets")

        graphList.extend(dataAugmentation(falseLabel, num - 1, rem, AUG_PERC))
        # print( num, k, len(feature_index), len(feature_index_choosen), len(data["featureMatrix"]))

    if len(unverifiedLabel) < maximum and len(unverifiedLabel) != 0:
        num = floor(maximum / len(unverifiedLabel))
        rem = maximum - (len(unverifiedLabel) * int(num))

        print(
            f"\nAugmenting entire Unverified labelled tweets {num - 1} and randomly audmenting {rem} tweets")

        graphList.extend(dataAugmentation(
            unverifiedLabel, num - 1, rem, AUG_PERC))

    pickle_path = os.path.join(augmented_save_path, event, 'graph.pickle')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n")
    print("*" * 60)

tk = TweetTokenizer()


def get_probability_for_tweets_selection(thread):
    probs = []
    for tweet in thread["featureMatrix"]:
        strToken = normalizeTweet(tweet[0])
        strToken = strToken.replace("@USER", "").replace("HTTPURL", "")
        strToken = tk.tokenize(strToken)
        probs.append(len(strToken))

    probs.pop(0)  # Remove source

    return probs


def nlpAugmentation(json_file, num, p_commets_aug=0.15):
    res = []
    with open(json_file, encoding="utf8") as ofile:
        data = json.load(ofile)

    # Extract 0.15% of comments to augment
    probs = get_probability_for_tweets_selection(data)

    k = int(ceil((len(data["featureMatrix"]) - 1) * p_commets_aug))
    feature_index = [i for i in range(1, len(data["featureMatrix"]))]

    if len(feature_index) != 0:

        feature_index_choosen = []
        for _ in range(num):
            feature_index_choosen.extend(
                random.choices(feature_index, weights=probs, k=k))

        tweets = [data["featureMatrix"][i][0] for i in feature_index_choosen]

        augmented_tweets = aug.augment(tweets)  # All the comments
        source_augmented_twetes = aug.augment(
            data["featureMatrix"][0][0], n=num)  # Source

        # Change augmented_tweets => 1xint(floor(len(data["featureMatrix"]) * 0.15)) * num => numxint(floor(len(data["featureMatrix"]) * 0.15))
        augmented_tweets = np.array(augmented_tweets).reshape(
            num, int(ceil((len(data["featureMatrix"]) - 1) * p_commets_aug)))
        feature_index_choosen = np.array(feature_index_choosen).reshape(
            num, int(ceil((len(data["featureMatrix"]) - 1) * p_commets_aug)))

        for f_idx, aug_source_tweet, aug_tweet in zip(feature_index_choosen, source_augmented_twetes, augmented_tweets,):
            tmp = copy.deepcopy(data)
            for i, j in zip(f_idx, aug_tweet):
                tmp["featureMatrix"][i][0] = j
            tmp["featureMatrix"][0][0] = aug_source_tweet
            res.append(getGData(data, tmp))

    else:
        source_augmented_twetes = aug.augment(
            data["featureMatrix"][0][0], n=num)  # Source
        for s in source_augmented_twetes:
            tmp = copy.deepcopy(data)
            tmp["featureMatrix"][0][0] = s
            res.append(getGData(data, tmp))

    return res


def dataAugmentation(data_list, num, rem, p_commets_aug=0.15):
    graphList = []
    if num >= 1:
        for tweet in data_list:
            augmented_data = nlpAugmentation(tweet, num, p_commets_aug)
            graphList.extend(augmented_data)

    rem_data = random.sample(data_list, rem)
    for tweet in rem_data:
        augmented_data = nlpAugmentation(tweet, 1, p_commets_aug)
        graphList.extend(augmented_data)

    return graphList


improved_data_save_path = os.path.join(SAVE_DIR, "improved")
os.makedirs(improved_data_save_path, exist_ok=True)
data = {}
for event in os.listdir(os.path.join(SAVE_DIR, "pheme")):
    print("\n")
    print("*" * 60)
    print(event)

    graphList = []

    trueLabel = [
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "true", i)
        for i in os.listdir(
            os.path.join(SAVE_DIR, "pheme", event, "rumours", "true"))]
    falseLabel = [os.path.join(SAVE_DIR, "pheme", event, "rumours", "false", i)for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "false"))]

    unverifiedLabel = [os.path.join(SAVE_DIR, "pheme", event, "rumours", "unverified", i)
                       for i in os.listdir(
        os.path.join(SAVE_DIR, "pheme", event, "rumours", "unverified"))]

    print(
        f"\nTrue : {len(trueLabel)} | False : {len(falseLabel)} | Unverified : {len(unverifiedLabel)} tweets\n")

    for x in [trueLabel, falseLabel, unverifiedLabel]:
        for f in x:
            gdata = {}
            with open(f, encoding="utf8") as ofile:
                data = json.load(ofile)

            edge_list = data["edgeList"]
            edge_list = np.array(edge_list).T.tolist()

            features = []
            tweetMatrix = data["tweetIDList"]
            for i, j in enumerate(data['featureMatrix']):
                local_features = TWEET_FEAT[tweetMatrix[i]]
                local_features.extend(j[1:])  # Social features
                features.append(local_features)

            gdata['x'] = features
            gdata['y'] = data["label"]
            gdata['edge_list'] = edge_list

            graphList.append(gdata)

    # Save the un-augmented graphList to be used as test dataset
    os.makedirs(os.path.join(improved_data_save_path, event), exist_ok=True)
    pickle_path = os.path.join(
        improved_data_save_path, event, 'graph_test.pickle')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)

    maximum = max(len(trueLabel), len(falseLabel), len(unverifiedLabel))

    if len(trueLabel) < maximum and len(trueLabel) != 0:
        num = floor(maximum / len(trueLabel))
        rem = maximum - (len(trueLabel) * int(num))

        print(
            f"\nAugmenting entire True labelled tweets {num - 1} and randomly audmenting {rem} tweets")

        graphList.extend(dataAugmentation(trueLabel, num - 1, rem, AUG_PERC))

    if len(falseLabel) < maximum and len(falseLabel) != 0:
        num = floor(maximum / len(falseLabel))
        rem = maximum - (len(falseLabel) * int(num))

        print(
            f"\nAugmenting entire False labelled tweets {num - 1} and randomly audmenting {rem} tweets")

        graphList.extend(dataAugmentation(falseLabel, num - 1, rem, AUG_PERC))
        # print( num, k, len(feature_index), len(feature_index_choosen), len(data["featureMatrix"]))

    if len(unverifiedLabel) < maximum and len(unverifiedLabel) != 0:
        num = floor(maximum / len(unverifiedLabel))
        rem = maximum - (len(unverifiedLabel) * int(num))

        print(
            f"\nAugmenting entire Unverified labelled tweets {num - 1} and randomly audmenting {rem} tweets")

        graphList.extend(dataAugmentation(
            unverifiedLabel, num - 1, rem, AUG_PERC))

    pickle_path = os.path.join(improved_data_save_path, event, 'graph.pickle')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n")
    print("*" * 60)
