from datetime import datetime
import numpy as np


def create_graph_with_map(structure_json, id, source_json, reaction_json):
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
                if comment not in combined_json.keys():
                    continue
                    
                nodeToIndexMap[comment] = currentIndex
                nodeToIndexMapArray.append(comment)
                edgeList.append((nodeToIndexMap[currentNode], currentIndex))
                currentIndex = currentIndex + 1
                stack.append((comment, context[comment]))
    del nodeToIndexMap['/']
    edgeList.pop(0)

    return edgeList, nodeToIndexMap, nodeToIndexMapArray


def get_features(tweet_json, feature_extractor):
    tweet_text = tweet_json["text"]
    user_feature = tweet_json["user"]
    social_features = feature_extractor.get_social_features(user_feature)
    return [tweet_text, *social_features]


def create_feature_matrix(source_json, reaction_json, nodeToIndexMap, nodeToIndexMapArray, feature_extractor):
    tweets = source_json
    tweets.update(reaction_json)
    feature_matrix = []
    tweet_id_list = []
    for tweet_id in nodeToIndexMapArray:
        feature_matrix.append(get_features(
            tweets[tweet_id], feature_extractor))
        tweet_id_list.append(tweet_id)
    return feature_matrix, tweet_id_list


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
                if comment not in combined_json.keys():
                    continue
                    
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
                if comment not in combined_json.keys():
                    continue
                    
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


def getGDataAugmented(tweet_data, augmented_data, feature_extractor):
    gdata = {}
    features = []
    for i, j in enumerate(augmented_data['featureMatrix']):
        local_features = []
        local_features.extend(
            feature_extractor.get_bert_features(j[0]))
        local_features.extend(j[1:])  # Social features
        features.append(local_features)

    gdata['x'] = features
    gdata['y'] = tweet_data['label']

    edge_list = tweet_data["edgeList"]
    edge_list = np.array(edge_list).T.tolist()
    gdata['edge_list'] = edge_list

    return gdata


def getGData(data, TWEET_FEAT):
    gdata = {}

    edge_list = data["edgeList"]
    edge_list = np.array(edge_list).T.tolist()

    features = []
    tweetMatrix = data["tweetIDList"]
    for i, j in enumerate(data['featureMatrix']):
        text_feat = TWEET_FEAT[tweetMatrix[i]]
        if len(text_feat) > 768:
            print(len(text_feat))
            raise "Error time 768"
        social_feat = j[1:]
        features.append([*text_feat, *social_feat])

        if len(features[-1]) > 772:
            print(len(features[-1]))
            raise "Error time 772"

    gdata['x'] = features
    gdata['y'] = data["label"]
    gdata['edge_list'] = edge_list

    return gdata
