from math import ceil, floor
import nlpaug.augmenter.word as naw
import json
import pickle
import numpy as np
import random
import os
from util import *
from util_graph import *
from feature_extractor import *
import copy
from tqdm import tqdm

aug = naw.ContextualWordEmbsAug(
    model_path='vinai/bertweet-base', aug_p=0.47, device="cuda", stopwords=["@USER", "HTTPURL"])


def nlpAugmentation(json_file, num, feat_extractor, p_commets_aug=0.47):
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
        feature_index_choosen.extend(np.random.choice(feature_index, k, replace=False).tolist())

    tweets = [data["featureMatrix"][i][0] for i in feature_index_choosen]

    augmented_tweets = aug.augment(tweets)  # All the comments
    source_augmented_twetes = aug.augment(data["featureMatrix"][0][0], n=num)  # Source
    if num == 1:
        source_augmented_twetes = [ source_augmented_twetes ]
        
    assert len(source_augmented_twetes) == num

    augmented_tweets = np.array(augmented_tweets).reshape(num, k)
    feature_index_choosen = np.array(feature_index_choosen).reshape(num, k)

    for f_idx, aug_source_tweet, aug_tweet in zip(feature_index_choosen, source_augmented_twetes, augmented_tweets):
        tmp = copy.deepcopy(data)
        for i, j in zip(f_idx, aug_tweet):
            tmp["featureMatrix"][i][0] = j
        tmp["featureMatrix"][0][0] = aug_source_tweet
        res.append(getGDataAugmented(data, tmp, feat_extractor))
    return res


def get_probability_for_tweets_selection(thread):
    tk = TweetTokenizer()
    probs = []
    for tweet in thread["featureMatrix"]:
        strToken = normalizeTweet(tweet[0])
        strToken = strToken.replace("@USER", "").replace("HTTPURL", "")
        strToken = tk.tokenize(strToken)
        probs.append(len(strToken))

    probs.pop(0)  # Remove source

    return probs


# +
# def nlpAugmentationImproved(json_file, num, feat_extractor, p_commets_aug=0.47):
#     res = []
#     with open(json_file, encoding="utf8") as ofile:
#         data = json.load(ofile)

#     # Extract 0.15% of comments to augment
#     probs = get_probability_for_tweets_selection(data)

#     k = int(ceil((len(data["featureMatrix"]) - 1) * p_commets_aug))
#     feature_index = [i for i in range(1, len(data["featureMatrix"]))]

#     if len(feature_index) != 0:
#         feature_index_choosen = []
#         for _ in range(num):
#             if sum(probs) != 0:
#                 feature_index_choosen.extend(
#                     random.choices(feature_index, weights=probs, k=k))
#             else:
#                 feature_index_choosen.extend(
#                     random.choices(feature_index, k=k))

#         tweets = [data["featureMatrix"][i][0] for i in feature_index_choosen]

#         augmented_tweets = aug.augment(tweets)  # All the comments
#         source_augmented_twetes = aug.augment(
#             data["featureMatrix"][0][0], n=num)  # Source

#         # Change augmented_tweets => 1xint(floor(len(data["featureMatrix"]) * 0.15)) * num => numxint(floor(len(data["featureMatrix"]) * 0.15))
#         augmented_tweets = np.array(augmented_tweets).reshape(
#             num, int(ceil((len(data["featureMatrix"]) - 1) * p_commets_aug)))
#         feature_index_choosen = np.array(feature_index_choosen).reshape(
#             num, int(ceil((len(data["featureMatrix"]) - 1) * p_commets_aug)))

#         for f_idx, aug_source_tweet, aug_tweet in zip(feature_index_choosen, source_augmented_twetes, augmented_tweets,):
#             tmp = copy.deepcopy(data)
#             for i, j in zip(f_idx, aug_tweet):
#                 tmp["featureMatrix"][i][0] = j
#             tmp["featureMatrix"][0][0] = aug_source_tweet
#             res.append(getGDataAugmented(data, tmp, feat_extractor))

#     else:
#         source_augmented_twetes = aug.augment(
#             data["featureMatrix"][0][0], n=num)  # Source
#         for s in source_augmented_twetes:
#             tmp = copy.deepcopy(data)
#             tmp["featureMatrix"][0][0] = s
#             res.append(getGDataAugmented(data, tmp, feat_extractor))

#     return res

# +
def nlpAugmentationImproved(json_file, num, feat_extractor, p_commets_aug=0.47):
    res = []
    with open(json_file, encoding="utf8") as ofile:
        data = json.load(ofile)

    # Extract 0.15% of comments to augment
    for i, j in enumerate(data["featureMatrix"]):
        # normalize the tweets before augmentation => Prevents <unk> due to hashtags, mentions and urls
        data["featureMatrix"][i][0] = normalizeTweet(j[0])
        
    # Extract 0.15% of comments to augment
    probs = get_probability_for_tweets_selection(data)

    feature_index = [i for i in range(1, len(data["featureMatrix"]))]

    k = int( ceil( len(feature_index) * p_commets_aug ) )
    
    if len(feature_index) != 0:
        
#         probs_with_index = [ [i, j] for i, j in zip( feature_index, probs )]
#         probs_with_index.sort(key=lambda x: x[1], reverse=True)
#         feature_index_choosen = [ i[0] for i in  probs_with_index[ : k ] ]
#         feature_index_choosen = feature_index_choosen * num
        
        feature_index_choosen = []
        for _ in range(num):
            if sum(probs) != 0:
                feature_index_choosen.extend(random.choices(feature_index, weights=probs, k=k))
            else:
                print("Got here!")
                feature_index_choosen.extend(random.choices(feature_index, k=k))

        tweets = [data["featureMatrix"][i][0] for i in feature_index_choosen]

        augmented_tweets = aug.augment(tweets)  # All the comments
        source_augmented_twetes = aug.augment(data["featureMatrix"][0][0], n=num)  # Source
        
        if num == 1:
            source_augmented_twetes = [ source_augmented_twetes ]

        assert len(source_augmented_twetes) == num
            
        # Change augmented_tweets => 1xint(floor(len(data["featureMatrix"]) * 0.15)) * num => numxint(floor(len(data["featureMatrix"]) * 0.15))
        augmented_tweets = np.array(augmented_tweets).reshape(num, k)
        feature_index_choosen = np.array(feature_index_choosen).reshape(num, k)

        for f_idx, aug_source_tweet, aug_tweet in zip(feature_index_choosen, source_augmented_twetes, augmented_tweets):
            tmp = copy.deepcopy(data)
            for i, j in zip(f_idx, aug_tweet):
                tmp["featureMatrix"][i][0] = j
            tmp["featureMatrix"][0][0] = aug_source_tweet
            res.append(getGDataAugmented(data, tmp, feat_extractor))

    else:
        source_augmented_twetes = aug.augment(data["featureMatrix"][0][0], n=num)  # Source
        
        if num == 1:
            source_augmented_twetes = [ source_augmented_twetes ]
            
        assert len(source_augmented_twetes) == num
            
        for s in source_augmented_twetes:
            tmp = copy.deepcopy(data)
            tmp["featureMatrix"][0][0] = s
            res.append(getGDataAugmented(data, tmp, feat_extractor))
            
    return res


# -

def dataAugmentation(data_list, num, rem, feat_extractor, p_commets_aug=0.47, improved=False):
    graphList = []
    if num >= 1:
        print("NUM: ", num)
        for tweet in tqdm(data_list):
            if improved == False:
                augmented_data = nlpAugmentation(
                    tweet, num, feat_extractor, p_commets_aug)
            else:
                augmented_data = nlpAugmentationImproved(
                    tweet, num, feat_extractor, p_commets_aug)

            graphList.extend(augmented_data)
    
    oldLenght = len(graphList)
    print( len(graphList) )
    rem_data = random.sample(data_list, rem)
    print("REM: ", rem)
    for tweet in tqdm(rem_data):
        if improved == False:
            augmented_data = nlpAugmentation(
                tweet, 1, feat_extractor, p_commets_aug)
        else:
            augmented_data = nlpAugmentationImproved(
                tweet, 1, feat_extractor, p_commets_aug)
        graphList.extend(augmented_data)
        
    print( len(graphList) - oldLenght )
    
    return graphList


def getAugmentedData3Label(SAVE_DIR, TWEET_FEAT, improved, feat_extractor, AUG_PERC=0.47):
    if improved:
        augmented_save_path = os.path.join(SAVE_DIR, "improved")
    else:
        augmented_save_path = os.path.join(SAVE_DIR, "augmented")

    os.makedirs(augmented_save_path, exist_ok=True)
    data = {}
    graphList = []
    for event in os.listdir(os.path.join(SAVE_DIR, "pheme")):
        print("\n")
        print("*" * 60)
        print(event)
        
        trueLabel = [
            os.path.join(SAVE_DIR, "pheme", event, "true", i)
            for i in os.listdir(
                os.path.join(SAVE_DIR, "pheme", event, "true"))]
        falseLabel = [os.path.join(SAVE_DIR, "pheme", event, "false", i)for i in os.listdir(
            os.path.join(SAVE_DIR, "pheme", event, "false"))]

        unverifiedLabel = [os.path.join(SAVE_DIR, "pheme", event, "unverified", i)
                           for i in os.listdir(
            os.path.join(SAVE_DIR, "pheme", event, "unverified"))]

        print(
            f"\nTrue : {len(trueLabel)} | False : {len(falseLabel)} | Unverified : {len(unverifiedLabel)} tweets\n")

        for x in [trueLabel, falseLabel, unverifiedLabel]:
            for f in x:
                with open(f, encoding="utf8") as ofile:
                    data = json.load(ofile)
                graphList.append(getGData(data, TWEET_FEAT))

        
        maximum = max(len(trueLabel), len(falseLabel), len(unverifiedLabel))

        if len(trueLabel) < maximum and len(trueLabel) != 0:
            num = floor(maximum / len(trueLabel))
            rem = maximum - (len(trueLabel) * int(num))

            print(
                f"\nAugmenting entire True labelled tweets {num - 1} and randomly audmenting {rem} tweets")

            graphList.extend(dataAugmentation(
                trueLabel, num - 1, rem, feat_extractor, AUG_PERC, improved))

        if len(falseLabel) < maximum and len(falseLabel) != 0:
            num = floor(maximum / len(falseLabel))
            rem = maximum - (len(falseLabel) * int(num))

            print(
                f"\nAugmenting entire False labelled tweets {num - 1} and randomly audmenting {rem} tweets")

            graphList.extend(dataAugmentation(
                falseLabel, num - 1, rem, feat_extractor, AUG_PERC, improved))
            # print( num, k, len(feature_index), len(feature_index_choosen), len(data["featureMatrix"]))

        if len(unverifiedLabel) < maximum and len(unverifiedLabel) != 0:
            num = floor(maximum / len(unverifiedLabel))
            rem = maximum - (len(unverifiedLabel) * int(num))

            print(
                f"\nAugmenting entire Unverified labelled tweets {num - 1} and randomly audmenting {rem} tweets")

            graphList.extend(dataAugmentation(
                unverifiedLabel, num - 1, rem, feat_extractor, AUG_PERC, improved))


        print("\n")
        print("*" * 60)
        
    pickle_path = os.path.join(augmented_save_path, 'graph.pickle')
    with open(pickle_path, 'wb') as handle:
        pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)


def getAugmentedData2Label(SAVE_DIR, TWEET_FEAT, improved, feat_extractor, AUG_PERC=0.47):
    data = {}

    if improved:
        improved_data_save_path = os.path.join(SAVE_DIR, "improved")
    else:
        improved_data_save_path = os.path.join(SAVE_DIR, "augmented")

    os.makedirs(improved_data_save_path, exist_ok=True)
    for event in os.listdir(os.path.join(SAVE_DIR, "pheme")):
        print("\n")
        print("*" * 60)
        print(event)

        graphList = []

        rumourLabel = [
            os.path.join(SAVE_DIR, "pheme", event, i) for i in os.listdir(os.path.join(SAVE_DIR, "pheme", event))]
        nonrumourLabel = [
            os.path.join(SAVE_DIR, "pheme", event, i) for i in os.listdir(os.path.join(SAVE_DIR, "pheme", event))]

        print(
            f"Rumour : {len(rumourLabel)} | Non-rumour : {len(nonrumourLabel)}")

        for x in [rumourLabel, nonrumourLabel]:
            for f in tqdm(x):
                with open(f, encoding="utf8") as ofile:
                    data = json.load(ofile)
                graphList.append(getGData(data, TWEET_FEAT))

        # Save the un-augmented graphList to be used as test dataset
        os.makedirs(os.path.join(
            improved_data_save_path, event), exist_ok=True)
        pickle_path = os.path.join(
            improved_data_save_path, event, 'graph_test.pickle')
        with open(pickle_path, 'wb') as handle:
            pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)

        maximum = max(len(rumourLabel), len(nonrumourLabel))

        if len(rumourLabel) < maximum and len(rumourLabel) != 0:
            num = floor(maximum / len(rumourLabel))
            rem = maximum - (len(rumourLabel) * int(num))

            print(
                f"\nAugmenting entire Rumour label tweets {num - 1} and randomly augmenting {rem} tweets")

            graphList.extend(dataAugmentation(
                rumourLabel, num - 1, rem, feat_extractor, AUG_PERC, improved))

        if len(nonrumourLabel) < maximum and len(nonrumourLabel) != 0:
            num = floor(maximum / len(nonrumourLabel))
            rem = maximum - (len(nonrumourLabel) * int(num))

            print(
                f"\nAugmenting entire Non-rumour label tweets {num - 1} and randomly augmenting {rem} tweets")

            graphList.extend(dataAugmentation(
                nonrumourLabel, num - 1, rem, feat_extractor, AUG_PERC, improved))

        pickle_path = os.path.join(
            improved_data_save_path, event, 'graph.pickle')
        with open(pickle_path, 'wb') as handle:
            pickle.dump(graphList, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("\n")
        print("*" * 60)
