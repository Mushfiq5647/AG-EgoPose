# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import os
import pickle
import json


class ValidationAnnotation(object):
    """ Simple vocabulary wrapper """

    def __init__(self):
        self.anns = []

    def __len__(self):
        return len(self.anns)


def build_validation_annotation():
    validation_annotation = ValidationAnnotation()
    validation_categories = {"patty5":2006,
                       "catch40": 1360,
                       "convo53":2323}

    for cat in validation_categories:
        count = 0
        count = validation_categories[cat]
        while (count - 256) > 0:
            validation_annotation.anns.append((cat, count))
            count -= 256
        validation_annotation.anns.append((cat, 257))
    return validation_annotation

def main(args):
    anns = build_validation_annotation()
    print("num sequences:", len(anns))
    validation_annotation_path = args.validation_annotation_path
    with open(validation_annotation_path, 'wb') as f:
        pickle.dump(anns, f)

    with open(validation_annotation_path, 'rb') as f:
        validation_annotations = pickle.load(f)

    annotations_dict = {
        "annotations": validation_annotations.anns
    }

    with open('validation_annotations.json', 'w') as f:
        json.dump(annotations_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_annotation_path', type=str, required=True, help='path to base dir for all captures')
    args = parser.parse_args()
    main(args)