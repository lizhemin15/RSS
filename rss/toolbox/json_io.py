import json


def load_json(path):
    with open(path, "r") as outfile:
        return json.load(outfile)


def save_json(path,dict_data):
    with open(path, "w") as outfile:
        json.dump(dict_data, outfile)