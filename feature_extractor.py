from emoji import demojize
import re
import torch
from nltk.tokenize import TweetTokenizer
from math import ceil, floor, log10
from transformers import AutoModel, AutoTokenizer


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
    tokenizer = TweetTokenizer()

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
