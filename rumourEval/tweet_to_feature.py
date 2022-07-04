from ast import ExtSlice
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

import sys

random.seed(12345)
np.random.seed(12345)

tokenizer = TweetTokenizer()

if len(sys.argv) >= 2:
    DATA_PATH = sys.argv[1]
else:
    DATA_PATH = "./rumoureval2019"

if len(sys.argv) >= 3:
    SAVE_DIR = sys.argv[2]
else:
    SAVE_DIR = "./data/"


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

args = {
    "path": DATA_PATH,
    "save_path": os.path.join(SAVE_DIR, "pheme")
}

# +
# Get the name of all the folders in the parent folder.

# load all training data
train_data_path = os.path.join( args["path"], "rumoureval-2019-training-data", "twitter-english")
directories = [os.path.join(train_data_path, o) for o in os.listdir(
    train_data_path) if os.path.isdir(os.path.join(train_data_path, o))]

# load all test data
test_data_path = os.path.join( args["path"], "rumoureval-2019-test-data", "twitter-en-test-data")
directories.extend( [os.path.join(test_data_path, o) for o in os.listdir(
    test_data_path) if os.path.isdir(os.path.join(test_data_path, o))] )
# -

print("*" * 60)
print("Number of Events: ", len(directories))
for idx, i in enumerate(directories):
    print(f"{(idx + 1)}. {PurePath(i).parts[-1].split('-')[0]}")
print("*" * 60)

# Traverse through it to get Source Tweet and Reaction Tweet
data = {}
for dir in directories:
    event = PurePath(dir).parts[-1].split('-')[0]
    print(f"Currently processing {event} event.")
    
    threads = [os.path.join(dir, i) for i in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, i))]

    for t in tqdm( threads ):
        id = PurePath(t).parts[-1]
        
        # Source folder
        source_path = os.path.join(t, "source-tweet", f"{id}.json")
        with open(source_path, encoding="utf8") as ofile:
            source_text = json.load(ofile)
            source_text = source_text["text"]
            data[id] = feature_extractor.get_bert_features(source_text)
            
        # Reaction Folder
        reaction_path = os.path.join(t, "replies")
        reactions = [os.path.join(reaction_path, i) for i in os.listdir(
            reaction_path) if i.split('\\')[-1][0] != '.']

        for reaction in reactions:
            with open(reaction, encoding="utf8") as ofile:
                obj = json.load(ofile)
                reaction_text = obj["text"]

            reaction_id = PurePath(reaction).parts[-1].split('.')[0]
            data[reaction_id] = feature_extractor.get_bert_features(
                reaction_text)

    print("\n")
    print("*" * 60)

os.makedirs(os.path.join(SAVE_DIR), exist_ok=True)
with open(os.path.join(SAVE_DIR, "tweet_features.pickle"), "wb") as handle:
    pickle.dump(data, handle)
