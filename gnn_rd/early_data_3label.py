import os
from pathlib import PurePath

from util import *
from util_graph import *


def getEarlyRDTime3Label(timeLimit, pheme_data, event, TWEET_FEAT, feature_extractor):
    # print(timiLimit, " Hour")
    timeLimit = timeLimit*3600
    return_list = []

    directories = [os.path.join(pheme_data, o) for o in os.listdir(
        pheme_data) if os.path.isdir(os.path.join(pheme_data, o))]

    for d in directories:
        if event in d:
            dir = d

    # Traverse through rumour and non-rumour directories
    sub_dir = [os.path.join(dir, i) for i in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, i))]

    for sdir in sub_dir:
        label = PurePath(sdir).parts[-1]
        if label == "non-rumours":
            continue

        tweets = [os.path.join(sdir, i) for i in os.listdir(
            sdir) if os.path.isdir(os.path.join(sdir, i))]

        for t in tweets:
            id = PurePath(t).parts[-1]

            # Structure
            structure_json = getStructure(t)
            # Annotations
            real_label = getAnnotation(t)
            if real_label == "non-rumours":
                raise "3 label encountered non-rumour label!"

            # Source folder
            source_json = getSource(t, id)

            # Reaction Folder
            reaction_json = getReactions(t)

            edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map_timeoffset(
                structure_json, id, source_json, reaction_json, timeLimit)
            featureMatrix, tweetMatrix = create_feature_matrix(
                source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray, feature_extractor)

            features = []
            for i, j in enumerate(featureMatrix):
                text_feat = TWEET_FEAT[tweetMatrix[i]]
                if len(text_feat) > 768:
                    print(len(text_feat))
                    raise "Error time 768"

                social_feat = j[1:]
                features.append([*text_feat, *social_feat])

                if len(features[-1]) > 772:
                    print(len(features[-1]))
                    raise "Error time 772"

            gdata = {}
            gdata['x'] = features
            gdata['y'] = real_label
            gdata['edge_list'] = np.array(edgeList).T.tolist()

            return_list.append(gdata)

    return return_list


def getEarlyRDComment3Label(commentLimit, pheme_data, event, TWEET_FEAT, feature_extractor):
    return_list = []

    directories = [os.path.join(pheme_data, o) for o in os.listdir(
        pheme_data) if os.path.isdir(os.path.join(pheme_data, o))]

    for d in directories:
        if event in d:
            dir = d

    # Traverse through rumour and non-rumour directories
    sub_dir = [os.path.join(dir, i) for i in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, i))]

    for sdir in sub_dir:
        label = PurePath(sdir).parts[-1]
        if label == "non-rumours":
            continue

        tweets = [os.path.join(sdir, i) for i in os.listdir(
            sdir) if os.path.isdir(os.path.join(sdir, i))]

        for t in tweets:
            id = PurePath(t).parts[-1]

            # Structure
            structure_json = getStructure(t)

            # Annotations
            real_label = getAnnotation(t)
            if real_label == "non-rumours":
                raise "3 label encountered non-rumour label!"

            # Source folder
            source_json = getSource(t, id)

            # Reaction Folder
            reaction_json = getReactions(t)

            edgeList, nodeToIndexMap, nodeToIndexMapArray = create_graph_with_map_comment(
                structure_json, id, source_json, reaction_json, commentLimit)
            featureMatrix, tweetMatrix = create_feature_matrix(
                source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray, feature_extractor)

            features = []
            for i, j in enumerate(featureMatrix):
                text_feat = TWEET_FEAT[tweetMatrix[i]]
                if len(text_feat) > 768:
                    print(len(text_feat))
                    raise "Error time 768"
                social_feat = j[1:]
                features.append([*text_feat, *social_feat])

                if len(features[-1]) > 772:
                    print(len(features[-1]))
                    raise "Error time 772"

            gdata = {}
            gdata['x'] = features
            gdata['y'] = real_label
            gdata['edge_list'] = np.array(edgeList).T.tolist()

            return_list.append(gdata)

    return return_list
