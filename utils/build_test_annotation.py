# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import os
import pickle
import json


class TestAnnotation(object):
    """ Simple vocabulary wrapper """

    def __init__(self):
        self.anns = []

    def __len__(self):
        return len(self.anns)


def build_test_annotation(dataset):
    test_annotation = TestAnnotation()
    kinect_test_categories = {"patty31":1410,
                       "patty32":2657,
                       "patty34":1787,
                       "patty35":1421,
                       "catch41": 1698,
                       "catch42": 2258,
                       "catch55":2257,
                       "convo54":2808,
                       "convo59":2887,
                       "sport57":3732}

    cmu_test_categories = {"3-catch3": 3395,
                           "7-convo4": 2767,
                           "8-convo5": 2736,
                           "9-convo6": 2753,
                           "11-hand2": 3453,
                           "12-hand3": 2454,
                           "14-sports2": 2133}
    if dataset == 'cmu':
        test_categories = cmu_test_categories
    else:
        categories = kinect_test_categories

    for cat in test_categories:
        count = 0
        count = test_categories[cat]
        while (count - 256) > 0:
            test_annotation.anns.append((cat, count))
            count -= 256
        test_annotation.anns.append((cat, 257))
    return test_annotation


def main(args):
    anns = build_test_annotation(args.dataset)
    print("num sequences:", len(anns))
    test_annotation_path = args.test_annotation_path
    with open(test_annotation_path, 'wb') as f:
        pickle.dump(anns, f)

    with open(test_annotation_path, 'rb') as f:
        test_annotations = pickle.load(f)

    annotations_dict = {
        "annotations": test_annotations.anns
    }

    with open('test_annotations.json', 'w') as f:
        json.dump(annotations_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_annotation_path', type=str, required=True, help='path to base dir for all captures')
    parser.add_argument('--dataset', type=str, required=True, help='kinect or cmu')
    args = parser.parse_args()
    main(args)

