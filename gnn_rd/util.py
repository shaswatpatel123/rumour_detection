import os
import json
from pathlib import PurePath


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


def getStructure(t):
    # Structure
    structure_path = os.path.join(t, 'structure.json')
    with open(structure_path, encoding="utf8") as ofile:
        content = ofile.read()
        clean = content.replace('”', '"')
        # json_data = json.loads(clean)
        structure_json = json.loads(clean)

    return structure_json


def getAnnotation(t):
    # Annotations
    annotation_path = os.path.join(t, 'annotation.json')
    with open(annotation_path, encoding="utf8") as ofile:
        annotation_json = json.load(ofile)
        label = convert_annotations(annotation_json)
        if label == "nonrumour":
            label = "non-rumours"

    return label


def getSource(t, id):
    source_json = {}
    if os.path.exists( os.path.join(t, "source-tweets" ) ):
        source_path = os.path.join(t, "source-tweets", f"{id}.json")
    else:
        source_path = os.path.join(t, "source-tweet", f"{id}.json")
        
    with open(source_path, encoding="utf8") as ofile:
        source_json[id] = json.load(ofile)

    return source_json


def getReactions(t):
    reaction_json = {}

    reaction_path = os.path.join(t, "reactions")
    reactions = [os.path.join(reaction_path, i) for i in os.listdir(
        reaction_path) if i.split('\\')[-1][0] != '.']

    for reaction in reactions:
        with open(reaction, encoding="utf8") as ofile:
            obj = json.load(ofile)
        reaction_id = PurePath(reaction).parts[-1].split('.')[0]

        reaction_json[reaction_id] = obj

    return reaction_json

def topicCheck( topic ):
    
    if topic in ["putinmissing", "prince", "gurlitt", "ebola"]:
        return False
    
    return True

